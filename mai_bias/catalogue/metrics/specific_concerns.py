import importlib

from mammoth_commons.datasets import Dataset
from mammoth_commons.models import Predictor
from mammoth_commons.exports import HTML, simplified_formatter
from mammoth_commons.reminders import logo_fairbench
from typing import List, Literal
from mammoth_commons.integration import metric
from mammoth_commons.externals import (
    fb_categories,
    align_predictions,
    CommonClassificationBenefits,
    compute_benefits,
)


@metric(
    namespace="mammotheu",
    version="v054",
    python="3.13",
    packages=("fairbench", "pandas", "onnxruntime", "ucimlrepo", "pygrank"),
    logo=logo_fairbench,
)
def specific_concerns(
    dataset: Dataset,
    model: Predictor,
    sensitive: List[str],
    intersections: Literal["Base", "All", "Subgroups"] = "Subgroups",
    base_measure: Literal[
        "Accuracy",
        "True positive rate",
        "True negative rate",
        "Area under curve",
        "Positive rate",
    ] = "Accuracy",
    compare_groups: Literal["Pairwise", "To the total population"] = "Pairwise",
    reduction: Literal[
        "Min",
        "Max",
        "Weighted mean",
        "Max difference",
        "Max relative difference",
        "Max betweeness area",
        "Standard deviation x2",
        "Gini coefficient",
    ] = "Max relative difference",
    problematic_deviation: float = 0.05,
    business_benefits: CommonClassificationBenefits = "Accuracy",
) -> HTML:
    """
    <h3>focus on a specific definition of fairness</h3>

    <p>Computes a fairness or bias measure that matches a specific type of numerical
    evaluation using the <a href="https://github.com/mever-team/FairBench">FairBench</a>
    library. The measure is built by combining simpler options to form more than 300 valid alternatives.</p>

    <span class="alert alert-warning alert-dismissible fade show" role="alert"
    style="display: inline-block; padding: 10px;"> <i class="bi bi-exclamation-triangle-fill"></i>
    This computes a specific fairness concerns and does not paint a broad enough picture. Make sure that
    you explore prospective biases with other modules first, like <i>model card</i>.</span>

    <details><summary><i>Technical details.</i></summary>

    <p>The assessment is conducted over sensitive attributes like gender, age, and race. Each attribute can have
    multiple values, such as several genders or races. Numeric attributes, like age, are normalized to the range [0,1]
    and treated as fuzzy values, where 0 indicates membership to a fuzzy group of "small" values, and 1 indicates
    membership to a fuzzy group of "large" values. A separate set of fairness metrics is calculated for each prediction
    label.</p>

    <p>If intersectional subgroup analysis is enabled, separate subgroups are created for each combination of sensitive
    attribute values. However, if there are too many attributes, some groups will be small or empty. Empty groups are
    ignored in the analysis.</p>
    </details>

    Args:
        intersections: Whether to consider only the provided groups (Base), all non-empty group intersections (All), or all non-empty intersections while ignoring larger groups during analysis (Subgroups). For example, the last option may not contain a `White` dimension if `White Men` is an existing dimension. This does nothing if there is only one sensitive attribute. It could be computationally intensive if too many group intersections are selected.
        base_measure: A base measure of algorithmic performance to be computed on each group.
        compare_groups: Whether to compare groups pairwise, or each group to the behavior of the whole population.
        reduction: The strategy with which to reduce all measure comparisons to one value.
        problematic_deviation: Sets up a threshold of when to consider deviation from ideal values as problematic. If nothing is considered problematic fairness is not necessarily achieved, but this is a good way to identify the most prominent biases. If value of 0 is set, all report values are shown, including those that have no ideal value.
        business_benefits: Which kind of business benefit does the model aim to maximize?
    """
    fb = importlib.import_module("fairbench")
    if isinstance(sensitive, str):
        sensitive = sensitive.split(",")
    assert len(sensitive) != 0, "At least one sensitive attribute should be selected"
    subtitle = "for sensitive attributes: <i>" + ", ".join(sensitive) + "</i>"
    predictions = model.predict(dataset, sensitive)
    dataset = dataset.to_csv(sensitive)
    sensitive = fb.Dimensions(
        {s: fb_categories(dataset.df[s]) for s in sensitive}, _separator=" "
    )
    if intersections != "Base":
        sensitive = sensitive.intersectional(delimiter=" - ")
    if intersections == "Subgroups":
        sensitive = sensitive.strict()
    assert (
        len(sensitive.branches()) != 0
    ), "Could not find any sensitive attribute intersections"
    predictions, labels = align_predictions(predictions, dataset.labels)
    predictions = predictions.columns
    labels = labels.columns if labels else None

    fb_measures = {
        "Accuracy": "acc",
        "True positive rate": "tpr",
        "True negative rate": "tnr",
        "Positive rate": "pr",
        "Area under curve": "auc",
    }
    vs_all = compare_groups != "vsall"
    fb_reductions = {
        "Min": "min",
        "Max": "max",
        "Weighted mean": "wmean",
        "Max difference": "maxdiff" if vs_all else "largestmaxdiff",
        "Max relative difference": "maxrel" if vs_all else "largestmaxrel",
        "Max betweeness area": "maxbarea" if vs_all else "largestmaxbarea",
        "Standard deviation x2": "stdx2",
        "Gini coefficient": "gini",
    }
    metric_name = (
        ("pairwise" if compare_groups == "Pairwise" else "vsall")
        + "_"
        + fb_reductions[reduction]
        + "_"
        + fb_measures[base_measure]
    )

    report = fb.quick.__getattr__(metric_name)(
        predictions=predictions, labels=labels, sensitive=sensitive
    )
    prob = float(problematic_deviation)
    assert 0 <= prob <= 1, "Problematic deviation should be in the range [0,1]"
    if prob:
        report = report.filter(fb.investigate.DeviationsOver(prob, prune=False))
    full_report = report.show(
        env=fb.export.Html(view=False, filename=None),
        depth=1 if isinstance(predictions, dict) else 0,
    )
    values = report.filter(fb.investigate.DeviationsOver(prob, prune=True))
    outcome = (
        "fair" if values.value is None and len(values.depends) == 0 else "biased"
    ) + f" {base_measure.lower()}"
    value = max([float(v) for v in report.flatten()])

    html_content = simplified_formatter(
        outcome=outcome.split(" ")[0],
        title_prefix="" if value < prob else f"{value*100:.0f}%",
        title=outcome + compute_benefits(business_benefits, predictions, labels),
        subtitle=subtitle,
        technology=logo_fairbench + "based on FairBench reporting",
        about=f"""
            We analysed how {getattr(fb.measures, fb_measures[base_measure]).descriptor.details.lower()} is 
            distributed in a model's outputs given a tested dataset by comparing several protected groups 
            {compare_groups.lower()}. 
            {'Expert interpretation of numeric details is required.' if prob == 0 else 
            'The assessment depends on specific parameters provided as inputs.'}
            """,
        methodology=f"""
            <p>The {reduction.lower()} of {getattr(fb.measures, fb_measures[base_measure]).descriptor.details.lower()} 
            is obtained across all protected groups, by comparing them {compare_groups.lower()}.
            The result is considered biased if it lays <b>{prob:.3f}</b> away from its ideal target 
            that would indicate fairness. For example, the ideal target is 0 for differences between measure values, 
            and 1 for values that should be large (e.g., the minimum accuracy across all groups).
            Some metrics have no known ideal values.</p>
            <p>The analysis considered <b>{len(sensitive.branches())}</b> protected groups:
            <br><i>{'<br>'.join(sensitive.branches().keys())}</i></p>
            """,
        pipeline=dataset.to_description(),
        experts=full_report.replace(metric_name, metric_name.replace("_", " ")),
    )
    return HTML(html_content)
