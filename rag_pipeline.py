"""
E-Commerce RAG Assistant with Query Routing & OpenTelemetry Tracing
===================================================================
Standalone modular implementation of the dual-index Weaviate RAG pipeline
for Fashion Forward Hub (44,423 clothing products + store FAQ collection).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import weaviate
from weaviate.classes.query import Filter
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name=os.getenv("PHOENIX_PROJECT_NAME", "ecommerce-rag-chatbot"),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"),
    )
    tracer = tracer_provider.get_tracer(__name__)
except Exception:
    tracer = trace.get_tracer(__name__)


def generate_params_dict(
    prompt: str,
    temperature: float = 1.0,
    role: str = "user",
    top_p: float = 1.0,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B",
) -> Dict[str, Any]:
    """Build the parameter dictionary for an OpenAI-compatible LLM call."""
    return {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model,
    }


def check_if_faq_or_product(
    query: str,
    llm_caller,
    simplified: bool = True,
) -> Tuple[str, int]:
    """
    Stage 1 Router: Classify a user query as 'FAQ' or 'Product'.
    When simplified=True, reduces router prompt tokens by ~69% (~203 -> ~62 tokens).
    """
    if not simplified:
        prompt = f"""Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
        Product related answers are answers specific about product information or that needs to use the products to give an answer.
        Examples:
                Is there a refund for incorrectly bought clothes? Label: FAQ
                Where are your stores located?: Label: FAQ
                Tell me about the cheapest T-shirts that you have. Label: Product
                Do you have blue T-shirts under 100 dollars? Label: Product
                What are the available sizes for the t-shirts? Label: FAQ
                How can I contact you via phone? Label: FAQ
                How can I find the promotions? Label: FAQ
                Give me ideas for a sunny look. Label: Product
        Return only one of the two labels: FAQ or Product, nothing more.
        Query to classify: {query}
                 """
    else:
        prompt = f"""Classify query as 'FAQ' (policies, sizing, contact, location) or 'Product' (specific items, outfits, prices).
Return exactly one word: FAQ or Product.
Query: {query}"""

    with tracer.start_as_current_span("routing_faq_or_product") as span:
        span.set_attribute("input.query", query)
        span.set_attribute("input.simplified", simplified)

        kwargs = generate_params_dict(prompt, temperature=0.0, max_tokens=10)
        with tracer.start_as_current_span("router_call") as router_span:
            response = llm_caller(**kwargs)
            router_span.set_attribute("llm.token_count.prompt", response["usage"]["prompt_tokens"])
            router_span.set_attribute("llm.token_count.completion", response["usage"]["completion_tokens"])
            router_span.set_attribute("llm.token_count.total", response["usage"]["total_tokens"])
            router_span.set_attribute("llm.model_name", response.get("model", kwargs["model"]))
            router_span.set_status(Status(StatusCode.OK))

        label = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]

        if "faq" in label.lower():
            label = "FAQ"
        elif "product" in label.lower():
            label = "Product"
        else:
            label = "undefined"

        span.set_attribute("output.label", label)
        span.set_attribute("output.total_tokens", total_tokens)
        span.set_status(Status(StatusCode.OK))
        return label, total_tokens


def generate_faq_layout(faq_dict: List[Dict[str, Any]]) -> str:
    """Format a list of FAQ entries into a structured text layout."""
    t = ""
    for f in faq_dict:
        t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"
    return t


def query_on_faq(
    query: str,
    faq_collection,
    full_faq_list: List[Dict[str, Any]],
    simplified: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Construct the FAQ response prompt.
    When simplified=True, performs top-5 semantic search in Weaviate (~65% fewer tokens).
    """
    if not simplified:
        with tracer.start_as_current_span("query_on_faq") as span:
            faq_layout = generate_faq_layout(full_faq_list)
            prompt = f"""You will be provided with an FAQ for a clothing store. 
        Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ. 
        <FAQ_ITEMS>
        PROVIDED FAQ: {faq_layout}
        </FAQ_ITEMS>
        Question: {query}
            """
            span.set_status(Status(StatusCode.OK))
            return generate_params_dict(prompt, **kwargs)
    else:
        with tracer.start_as_current_span("query_on_faq") as span:
            with tracer.start_as_current_span("retrieve_faq_questions") as retrieve_span:
                results = faq_collection.query.near_text(query, limit=5)
                for i, document in enumerate(results.objects):
                    retrieve_span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
                    retrieve_span.set_attribute(f"retrieval.documents.{i}.document.content", str(document.properties))
                items = [x.properties for x in results.objects]
                items.reverse()
                faq_layout = generate_faq_layout(items)

            prompt = (
                f"You will be provided with a query for a clothing store regarding FAQ. It will be provided relevant FAQ from the clothing store."
                f"Answer the query based on the relevant FAQ provided. They are ordered in decreasing relevance, so the first is the most relevant FAQ and the last is the least relevant."
                f"Answer the instruction based on them. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.\n"
                f"<FAQ>\n"
                f"RELEVANT FAQ ITEMS:\n{faq_layout}\n"
                f"</FAQ>\n"
                f"Query: {query}"
            )
            span.set_status(Status(StatusCode.OK))
            return generate_params_dict(prompt, **kwargs)


def decide_task_nature(
    query: str,
    llm_caller,
    simplified: bool = True,
) -> Tuple[str, int]:
    """
    Stage 2 Router: Classify a product query as 'creative' (styling/outfits)
    or 'technical' (catalog/prices).
    """
    if not simplified:
        prompt = f"""Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices etc.). Label it as creative or technical.
          Examples:
          Give me suggestions on a nice look for a nightclub. Label: creative
          What are the blue dresses you have available? Label: technical
          Give me three Tshirts for summer. Label: technical
          Give me a look for attending a wedding party. Label: creative
          Give me suggestions on clothes that match a green Tshirt. Label: creative
          I would like a suggestion on which products match a green Tshirt I already have. Label: creative

          Query to be analyzed: {query}. Only output one token with the label
          """
    else:
        prompt = f"""Classify query as 'creative' (making outfits/suggestions) or 'technical' (prices/availability).
Return only one word: creative or technical.
Query: {query}"""

    with tracer.start_as_current_span("decide_task_nature") as span:
        kwargs = generate_params_dict(prompt, temperature=0.0, max_tokens=1)
        response = llm_caller(**kwargs)
        label = response["choices"][0]["message"]["content"].strip().lower()
        total_tokens = response["usage"]["total_tokens"]
        span.set_attribute("llm.token_count.total", total_tokens)
        span.set_status(Status(StatusCode.OK))
        return label, total_tokens


def get_params_for_task(task: str) -> Dict[str, float]:
    """Return tailored temperature and top_p settings for creative vs. technical tasks."""
    parameters_dict = {
        "creative": {"top_p": 0.9, "temperature": 1.0},
        "technical": {"top_p": 0.7, "temperature": 0.3},
    }
    return parameters_dict.get(task, {"top_p": 0.5, "temperature": 1.0})


def parse_json_output(llm_output: str) -> Optional[Dict[str, Any]]:
    """Clean and parse an LLM-generated JSON string into a Python dictionary."""
    try:
        cleaned = (
            llm_output.replace("\n", "")
            .replace("'", "")
            .replace("}}", "}")
            .replace("{{", "{")
        )
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def get_filter_by_metadata(json_output: Optional[Dict[str, Any]] = None) -> Optional[List[Any]]:
    """Convert a parsed metadata dictionary into a list of Weaviate Filter objects."""
    if json_output is None:
        return None

    valid_keys = (
        "gender",
        "masterCategory",
        "articleType",
        "baseColour",
        "price",
        "usage",
        "season",
    )
    filters = []
    for key, value in json_output.items():
        if key not in valid_keys:
            continue
        if key == "price":
            if not isinstance(value, dict):
                continue
            min_price = value.get("min")
            max_price = value.get("max")
            if min_price is None or max_price is None:
                continue
            if min_price <= 0 or max_price == "inf":
                continue
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        else:
            filters.append(Filter.by_property(key).contains_any(value))
    return filters


def generate_items_context(results: List[Any]) -> str:
    """Format retrieved Weaviate product objects into a context block."""
    t = ""
    for item in results:
        props = item.properties
        t += (
            f"Product ID: {props['product_id']}. "
            f"Product name: {props['productDisplayName']}. "
            f"Product Category: {props['masterCategory']}. "
            f"Product usage: {props['usage']}. "
            f"Product gender: {props['gender']}. "
            f"Product Type: {props['articleType']}. "
            f"Product Category: {props['subCategory']} "
            f"Product Color: {props['baseColour']}. "
            f"Product Season: {props['season']}. "
            f"Product Year: {props['year']}.\n"
        )
    return t
