from mammoth import testing
from catalogue.model_loaders.onnx_ensemble import model_onnx_ensemble
from catalogue.dataset_loaders.uci_csv import data_uci
from catalogue.metrics.Multi_objective_report import Multi_objective_report


def test_multiobjective_report():
    with testing.Env(model_onnx_ensemble, Multi_objective_report, data_uci) as env:
        dataset_name = "credit"
        target = "Y"
        dataset = env.data_uci(dataset_name=dataset_name, target=target)
        model_path = "data/credit_mfppb.zip"
        model = env.model_onnx_ensemble(model_path)
        sensitive = ["X2", "X4", "X5"]
        html_result = env.Multi_objective_report(dataset, model, sensitive=sensitive)
        html_result.show()


if __name__ == "__main__":
    test_multiobjective_report()
