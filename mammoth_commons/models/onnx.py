import numpy as np
from mammoth_commons.models.predictor import Predictor


class ONNX(Predictor):
    def __init__(self, model_bytes, includes_sensitive=False):
        super().__init__()
        self.model_bytes = model_bytes
        self.includes_sensitive = includes_sensitive

    def predict(self, dataset, sensitive: list[str]):
        includes_sensitive = not self.includes_sensitive

        import onnxruntime as rt
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

        sess = rt.InferenceSession(self.model_bytes, providers=["CPUExecutionProvider"])
        label_name = sess.get_outputs()[0].name
        inputs = sess.get_inputs()
        input_count = len(inputs)

        if input_count == 1:
            x = (
                dataset
                if isinstance(dataset, np.ndarray)
                else dataset.to_pred(sensitive if includes_sensitive else list())
            )
            onnx_type_to_np = {
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(int32)": np.int32,
                "tensor(int64)": np.int64,
            }
            onnx_type = inputs[0].type
            np_type = onnx_type_to_np.get(onnx_type, None)
            assert np_type, f"ONNX model expects an unsupported input type: {onnx_type}"
            x = x.astype(np_type)
            inp = inputs[0]
            assert len(inp.shape) == 2, "ONNX model expects a 2D input"
            expected_cols = (
                inp.shape[1] if isinstance(inp.shape[1], int) else x.shape[1]
            )
            assert expected_cols == x.shape[1], (
                f"ONNX model expects {expected_cols} columns but dataset has {x.shape[1]}. "
                f"Sensitive attributes mismatch? Input name={inp.name}"
            )
            feed = {inp.name: x}
        else:
            # convert dataset to csv format to get the underlying dataframe
            # note: we do need the dataset itself externally as it's the standardization assumed by metrics
            dataset = dataset.to_csv(None)
            df = dataset.to_csv(None).df
            df = df[
                dataset.num + dataset.cat
            ]  # .df was the raw csv dataset so re-filter some stuff
            if includes_sensitive:
                df = df.drop(columns=sensitive, errors="ignore")
            onnx_inputs = [inp.name for inp in inputs]
            missing = [c for c in onnx_inputs if c not in df.columns]
            assert (
                not missing
            ), f"The dataset is missing required columns for the ONNX model: {missing}"
            x = df[onnx_inputs].to_numpy()
            assert x.ndim == 2, f"Dataset must be 2D, got shape {x.shape}"
            assert (
                x.shape[1] == input_count
            ), f"ONNX model expects {input_count} input columns but dataset has {x.shape[1]}"
            onnx_type_to_np = {
                "tensor(float)": np.float32,
                "tensor(double)": np.float64,
                "tensor(int32)": np.int32,
                "tensor(int64)": np.int64,
            }
            feed = {}
            for col_idx, inp in enumerate(inputs):
                onnx_type = inp.type
                if onnx_type == "tensor(string)":
                    # ONNX Runtime requires object-dtype numpy arrays for strings
                    feed[inp.name] = (
                        df[inp.name]
                        .astype(str)
                        .apply(lambda x: x.encode("utf-8"))  # ORT likes bytes
                        .to_numpy()
                        .reshape(-1, 1)
                        .astype(object)
                    )
                else:
                    np_type = onnx_type_to_np.get(onnx_type, None)
                    assert np_type, f"Unsupported ONNX input type: {onnx_type}"
                    feed[inp.name] = (
                        df[inp.name].to_numpy().reshape(-1, 1).astype(np_type)
                    )

        try:
            ret = sess.run([label_name], feed)[0]
            return ret
        except InvalidArgument as e:
            raise Exception(
                "The ONNX loader encountered an error matching this dataset with the model.<br><br>"
                '<details><summary class="btn btn-secondary">Details</summary><br><br>'
                "<pre>" + str(e) + "</pre></details>"
            )
