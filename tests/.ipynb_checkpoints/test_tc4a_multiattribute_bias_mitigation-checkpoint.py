'''
This test loads a mmm-fair classifier to the onnx_ensemble module.
To create the loadable .zip file e.g. credit_mfppb.zip, please refer to:
https://pypi.org/project/mmm-fair/
'''
from mammoth import testing
from catalogue.model_loaders.onnx_ensemble import model_onnx_ensemble
from catalogue.dataset_loaders.uci_csv import data_uci
from catalogue.metrics.model_card import model_card


def test_multiattribute_bias_mitigation():
    with testing.Env(model_onnx_ensemble, data_uci, model_card) as env:
        dataset_name = "credit"
        target = "Y"
        dataset = env.data_uci(dataset_name=dataset_name, target=target)
        model_path = "data/credit_mfppb.zip"
        model = env.model_onnx_ensemble(model_path)
        sensitive = ["X2", "X4", "X5"]
        markdown_result = env.model_card(dataset, model, sensitive=sensitive)
        markdown_result.show()


if __name__ == "__main__":
    test_multiattribute_bias_mitigation()
