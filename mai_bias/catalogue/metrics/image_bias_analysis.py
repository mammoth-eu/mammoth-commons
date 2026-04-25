from mammoth_commons.datasets import ImageLike
from mammoth_commons.models import EmptyModel
from mammoth_commons.exports import HTML, Markdown, simplified_formatter
from typing import List, Literal
from mammoth_commons.integration import metric, Options
from mammoth_commons.reminders import logo_vbmitigator


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("torch", "torchvision", "cvbiasmitigation"),
    logo=logo_vbmitigator,
)
def image_bias_analysis(
    dataset: ImageLike,
    model: EmptyModel,
    sensitive: List[str],
    task: Literal["face verification", "image classification"] = None,
) -> HTML:
    """
    <h3>for data scientists: solutions for imbalanced image models</h3>

    This module provides a comprehensive solution for analyzing image bias and recommending effective
    mitigation strategies. It can be used for both classification tasks (e.g., facial attribute
    extraction) and face verification. The core functionality revolves around evaluating how well
    different population groups, defined by a given protected attribute (such as gender, age, or
    ethnicity), are represented in the dataset. Representation bias occurs when some groups are
    overrepresented or underrepresented, leading to models that may perform poorly or unfairly on
    certain groups.

    Additionally, the module detects spurious correlations between the target attribute (e.g., the
    label a model is trying to predict) and other annotated attributes (such as image features like
    color or shape). Spurious correlations are misleading patterns that do not reflect meaningful
    relationships and can cause a model to make biased or inaccurate predictions. By identifying and
    addressing these hidden biases, the module helps improve the fairness and accuracy of your model.

    When you run the analysis, the module identifies specific biases within the dataset and suggests
    tailored mitigation approaches. Specifically, the suitable mitigation methodologies are determined
    based on the task and the types of the detected biases in the data.
    The analysis is conducted based on the
    <a href="https://github.com/gsarridis/cv-bias-mitigation-library">CV bias mitigation library</a>.

    Args:
        task: The type of predictive task. It should be either face verification or image classification.
    """
    from cvbiasmitigation.suggest import analysis

    assert task in [
        "face verification",
        "image classification",
    ], "The provided task should be either face verification or image classification"
    assert sensitive, "No sensitive attributes provided"
    if isinstance(sensitive, str):
        sensitive = [sens.strip() for sens in sensitive.split(",")]
    subtitle = "for sensitive attributes: <i>" + ", ".join(sensitive) + "</i>"
    json, title = analysis(
        dataset.path, task, dataset.target, sensitive, output="json", return_title=True
    )

    info = {}

    def json_to_str_recursively(
        data, indent=0, pending_close=[""]
    ):  # this is intentionally modifiable
        """Converts a JSON object to a string iteratively and recursively."""
        result_str = ""
        if isinstance(data, list):
            for item in data:
                result_str += json_to_str_recursively(item, indent)
            if pending_close[0]:
                result_str += pending_close[0]
                pending_close[0] = ""
        elif isinstance(data, dict):
            if data.get("type") == "heading":
                level = data["level"]
                if pending_close[0]:
                    result_str += pending_close[0]
                    pending_close[0] = ""
                if "title" not in info:
                    info["title"] = data["content"]
                if level == 1:
                    result_str += f"# {data['content']}\n"
                elif level == 2:
                    result_str += f"## {data['content'].capitalize()}\n"
                elif level == 3:
                    result_str += f"\n\n**{data['content'].capitalize()}**\n\n"
                elif level == 4:
                    result_str += f"\n\n*{data['content'].capitalize()}.* "
                elif level == 5:
                    result_str += f"\n\n<details> <summary>{data['content'].capitalize()}</summary>"
                    pending_close[0] += "\n\n</details>\n\n"
                else:
                    result_str += f"\n\n*{data['content'].capitalize()}.* "
            elif data.get("type") == "paragraph":
                content = data.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            result_str += " " * indent + item.get("content", "")
                        elif item.get("type") == "inline_code":
                            result_str += f" <span style='font-size:small;'>{item.get('content', '')}</span> "
                        elif item.get("type") == "link":
                            result_str += (
                                f"[{item.get('content', '')}]({item.get('url', '')})"
                            )
                        elif item.get("type") == "code":
                            result_str += (
                                "\n"
                                + " " * (indent + 4)
                                + "```"
                                + item.get("language", "")
                                + "\n"
                            )
                            result_str += (
                                " " * (indent + 4) + item.get("content", "") + "\n"
                            )
                            result_str += " " * (indent + 4) + "```"
                else:
                    result_str += " " * indent + str(content)
                if "about" not in info:
                    info["about"] = result_str
                result_str += "\n"  # newline after paragraph
            elif data.get("type") == "code":
                result_str += " " * indent + "```" + data.get("language", "") + "\n"
                result_str += " " * indent + data.get("content", "") + "\n"
                result_str += " " * indent + "```\n"
            elif data.get("type") == "html":
                result_str += " " * indent + data.get("content", "") + "\n"
            elif data.get("type") == "list":
                result_str += json_to_str_recursively(data.get("content", []), indent)
            else:
                result_str += " " * indent + str(data) + "\n"
        else:
            result_str += " " * indent + str(data) + "\n"

        return result_str

    output_str = json_to_str_recursively(json)
    title = info["title"]  # important: after population by json to str

    html_content = simplified_formatter(
        outcome="biased" if "bias" in title else "fair",
        technology=logo_vbmitigator + "based on vb-mitigator",
        title=title,
        subtitle=subtitle,
        about=info["about"],
        methodology=f"""
            <p>We used the <a href="https://github.com/gsarridis/cv-bias-mitigation-library">CV bias mitigation library</a>
            to detect representation imbalances, as well as spurious correlations between the target attribute (e.g., the
            label a model is trying to predict) and other annotated attributes (such as image features like
            color or shape). Spurious correlations are misleading patterns that do not reflect meaningful
            relationships and can cause a model to make biased or inaccurate predictions.
            Means of training fairer models are discussed in the <i>for experts</i> section.</p>
            """,
        pipeline=f"{dataset.to_description()}<br><br>{model.to_description()}",
        experts=Markdown(output_str).text(),
    )
    return HTML(html_content)
