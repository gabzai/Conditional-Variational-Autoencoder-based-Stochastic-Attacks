
�[root"_tf_keras_network*�[{"name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "orthonormal_basis"}, "name": "orthonormal_basis", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "base_t0", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t0", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "base_t1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t1", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "base_t2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t2", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "psiPhi_t0", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t0", "inbound_nodes": [[["base_t0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "psiPhi_t1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t1", "inbound_nodes": [[["base_t1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "psiPhi_t2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t2", "inbound_nodes": [[["base_t2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "v"}, "name": "v", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "psi_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "psi_layer", "inbound_nodes": [[["psiPhi_t0", 0, 0, {}], ["psiPhi_t1", 0, 0, {}], ["psiPhi_t2", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "synthetic_trace", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAMAAABTAAAAcxAAAAB8AGQBGQB8AGQCGQAXAFMAKQNO6QAAAADpAQAAAKkA\nKQHaAXhyAwAAAHIDAAAA+kcvaG9tZS96YWlkZy9CdXJlYXUvR1pBL0dpdEh1Yi9hcnRpY2xlX3Zh\nZXN0L2NvZGUvY3ZhZXN0X2FyY2hpdGVjdHVyZS5wedoIPGxhbWJkYT6lAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "synthetic_trace", "inbound_nodes": [[["v", 0, 0, {}], ["psi_layer", 0, 0, {}]]]}], "input_layers": [["v", 0, 0], ["orthonormal_basis", 0, 0]], "output_layers": [["synthetic_trace", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 256]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 3]}, {"class_name": "TensorShape", "items": [null, 3, 256]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3]}, "float32", "v"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 256]}, "float32", "orthonormal_basis"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3]}, "float32", "v"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 256]}, "float32", "orthonormal_basis"]}], "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "orthonormal_basis"}, "name": "orthonormal_basis", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Lambda", "config": {"name": "base_t0", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t0", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Lambda", "config": {"name": "base_t1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t1", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Lambda", "config": {"name": "base_t2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "base_t2", "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "psiPhi_t0", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t0", "inbound_nodes": [[["base_t0", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "psiPhi_t1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t1", "inbound_nodes": [[["base_t1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "psiPhi_t2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "psiPhi_t2", "inbound_nodes": [[["base_t2", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "v"}, "name": "v", "inbound_nodes": [], "shared_object_id": 13}, {"class_name": "Concatenate", "config": {"name": "psi_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "psi_layer", "inbound_nodes": [[["psiPhi_t0", 0, 0, {}], ["psiPhi_t1", 0, 0, {}], ["psiPhi_t2", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Lambda", "config": {"name": "synthetic_trace", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAMAAABTAAAAcxAAAAB8AGQBGQB8AGQCGQAXAFMAKQNO6QAAAADpAQAAAKkA\nKQHaAXhyAwAAAHIDAAAA+kcvaG9tZS96YWlkZy9CdXJlYXUvR1pBL0dpdEh1Yi9hcnRpY2xlX3Zh\nZXN0L2NvZGUvY3ZhZXN0X2FyY2hpdGVjdHVyZS5wedoIPGxhbWJkYT6lAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "synthetic_trace", "inbound_nodes": [[["v", 0, 0, {}], ["psi_layer", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["v", 0, 0], ["orthonormal_basis", 0, 0]], "output_layers": [["synthetic_trace", 0, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "orthonormal_basis", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 256]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "orthonormal_basis"}}2
�root.layer-1"_tf_keras_layer*�{"name": "base_t0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "base_t0", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 1}2
�root.layer-2"_tf_keras_layer*�{"name": "base_t1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "base_t1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 2}2
�root.layer-3"_tf_keras_layer*�{"name": "base_t2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "base_t2", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAAATAAAAcxYAAAB8AGQAZACFAogAZABkAIUCZgMZAFMAKQFOqQApAdoB\neCkB2gFpcgEAAAD6Ry9ob21lL3phaWRnL0J1cmVhdS9HWkEvR2l0SHViL2FydGljbGVfdmFlc3Qv\nY29kZS9jdmFlc3RfYXJjaGl0ZWN0dXJlLnB52gg8bGFtYmRhPqEAAADzAAAAAA==\n", null, {"class_name": "__tuple__", "items": [2]}]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["orthonormal_basis", 0, 0, {}]]], "shared_object_id": 3}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "psiPhi_t0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "psiPhi_t0", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["base_t0", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "psiPhi_t1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "psiPhi_t1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["base_t1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "psiPhi_t2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "psiPhi_t2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["base_t2", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}2
�root.layer-7"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "v", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "v"}}2
�	root.layer-8"_tf_keras_layer*�{"name": "psi_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "psi_layer", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["psiPhi_t0", 0, 0, {}], ["psiPhi_t1", 0, 0, {}], ["psiPhi_t2", 0, 0, {}]]], "shared_object_id": 14, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}2
�
root.layer-9"_tf_keras_layer*�{"name": "synthetic_trace", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "synthetic_trace", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAMAAABTAAAAcxAAAAB8AGQBGQB8AGQCGQAXAFMAKQNO6QAAAADpAQAAAKkA\nKQHaAXhyAwAAAHIDAAAA+kcvaG9tZS96YWlkZy9CdXJlYXUvR1pBL0dpdEh1Yi9hcnRpY2xlX3Zh\nZXN0L2NvZGUvY3ZhZXN0X2FyY2hpdGVjdHVyZS5wedoIPGxhbWJkYT6lAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "cvaest_architecture", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "inbound_nodes": [[["v", 0, 0, {}], ["psi_layer", 0, 0, {}]]], "shared_object_id": 15}2