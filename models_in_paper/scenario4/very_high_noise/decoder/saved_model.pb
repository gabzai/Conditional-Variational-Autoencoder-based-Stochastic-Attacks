ؐ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.22v2.6.1-9-gc2363d6d0258��
}
psiPhi_t0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namepsiPhi_t0/kernel
v
$psiPhi_t0/kernel/Read/ReadVariableOpReadVariableOppsiPhi_t0/kernel*
_output_shapes
:	�*
dtype0
t
psiPhi_t0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepsiPhi_t0/bias
m
"psiPhi_t0/bias/Read/ReadVariableOpReadVariableOppsiPhi_t0/bias*
_output_shapes
:*
dtype0
}
psiPhi_t1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namepsiPhi_t1/kernel
v
$psiPhi_t1/kernel/Read/ReadVariableOpReadVariableOppsiPhi_t1/kernel*
_output_shapes
:	�*
dtype0
t
psiPhi_t1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepsiPhi_t1/bias
m
"psiPhi_t1/bias/Read/ReadVariableOpReadVariableOppsiPhi_t1/bias*
_output_shapes
:*
dtype0
}
psiPhi_t2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namepsiPhi_t2/kernel
v
$psiPhi_t2/kernel/Read/ReadVariableOpReadVariableOppsiPhi_t2/kernel*
_output_shapes
:	�*
dtype0
t
psiPhi_t2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namepsiPhi_t2/bias
m
"psiPhi_t2/bias/Read/ReadVariableOpReadVariableOppsiPhi_t2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
 
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
 
*
0
1
"2
#3
(4
)5
*
0
1
"2
#3
(4
)5
�
regularization_losses

6layers
7layer_metrics
	variables
8layer_regularization_losses
trainable_variables
9non_trainable_variables
:metrics
 
 
 
 
�
regularization_losses

;layers
<layer_metrics
	variables
=layer_regularization_losses
trainable_variables
>non_trainable_variables
?metrics
 
 
 
�
regularization_losses

@layers
Alayer_metrics
	variables
Blayer_regularization_losses
trainable_variables
Cnon_trainable_variables
Dmetrics
 
 
 
�
regularization_losses

Elayers
Flayer_metrics
	variables
Glayer_regularization_losses
trainable_variables
Hnon_trainable_variables
Imetrics
\Z
VARIABLE_VALUEpsiPhi_t0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEpsiPhi_t0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses

Jlayers
Klayer_metrics
	variables
Llayer_regularization_losses
 trainable_variables
Mnon_trainable_variables
Nmetrics
\Z
VARIABLE_VALUEpsiPhi_t1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEpsiPhi_t1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
�
$regularization_losses

Olayers
Player_metrics
%	variables
Qlayer_regularization_losses
&trainable_variables
Rnon_trainable_variables
Smetrics
\Z
VARIABLE_VALUEpsiPhi_t2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEpsiPhi_t2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
*regularization_losses

Tlayers
Ulayer_metrics
+	variables
Vlayer_regularization_losses
,trainable_variables
Wnon_trainable_variables
Xmetrics
 
 
 
�
.regularization_losses

Ylayers
Zlayer_metrics
/	variables
[layer_regularization_losses
0trainable_variables
\non_trainable_variables
]metrics
 
 
 
�
2regularization_losses

^layers
_layer_metrics
3	variables
`layer_regularization_losses
4trainable_variables
anon_trainable_variables
bmetrics
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
!serving_default_orthonormal_basisPlaceholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
t
serving_default_vPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall!serving_default_orthonormal_basisserving_default_vpsiPhi_t0/kernelpsiPhi_t0/biaspsiPhi_t1/kernelpsiPhi_t1/biaspsiPhi_t2/kernelpsiPhi_t2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_50039
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$psiPhi_t0/kernel/Read/ReadVariableOp"psiPhi_t0/bias/Read/ReadVariableOp$psiPhi_t1/kernel/Read/ReadVariableOp"psiPhi_t1/bias/Read/ReadVariableOp$psiPhi_t2/kernel/Read/ReadVariableOp"psiPhi_t2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_50367
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamepsiPhi_t0/kernelpsiPhi_t0/biaspsiPhi_t1/kernelpsiPhi_t1/biaspsiPhi_t2/kernelpsiPhi_t2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_50395��
�

�
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_50258

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference__traced_save_50367
file_prefix/
+savev2_psiphi_t0_kernel_read_readvariableop-
)savev2_psiphi_t0_bias_read_readvariableop/
+savev2_psiphi_t1_kernel_read_readvariableop-
)savev2_psiphi_t1_bias_read_readvariableop/
+savev2_psiphi_t2_kernel_read_readvariableop-
)savev2_psiphi_t2_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_psiphi_t0_kernel_read_readvariableop)savev2_psiphi_t0_bias_read_readvariableop+savev2_psiphi_t1_kernel_read_readvariableop)savev2_psiphi_t1_bias_read_readvariableop+savev2_psiphi_t2_kernel_read_readvariableop)savev2_psiphi_t2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*J
_input_shapes9
7: :	�::	�::	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
^
B__inference_base_t0_layer_call_and_return_conditional_losses_49692

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_base_t2_layer_call_fn_50229

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_498872
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_49704

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_decoder_layer_call_fn_50133
inputs_0
inputs_1
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_497612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
^
B__inference_base_t2_layer_call_and_return_conditional_losses_50211

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_psiPhi_t2_layer_call_fn_50286

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_497362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_base_t0_layer_call_and_return_conditional_losses_50159

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
v
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50313
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
^
B__inference_base_t2_layer_call_and_return_conditional_losses_50219

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
B__inference_decoder_layer_call_and_return_conditional_losses_49994
v
orthonormal_basis"
psiphi_t0_49976:	�
psiphi_t0_49978:"
psiphi_t1_49981:	�
psiphi_t1_49983:"
psiphi_t2_49986:	�
psiphi_t2_49988:
identity��!psiPhi_t0/StatefulPartitionedCall�!psiPhi_t1/StatefulPartitionedCall�!psiPhi_t2/StatefulPartitionedCall�
base_t2/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_496722
base_t2/PartitionedCall�
base_t1/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_496822
base_t1/PartitionedCall�
base_t0/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_496922
base_t0/PartitionedCall�
!psiPhi_t0/StatefulPartitionedCallStatefulPartitionedCall base_t0/PartitionedCall:output:0psiphi_t0_49976psiphi_t0_49978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_497042#
!psiPhi_t0/StatefulPartitionedCall�
!psiPhi_t1/StatefulPartitionedCallStatefulPartitionedCall base_t1/PartitionedCall:output:0psiphi_t1_49981psiphi_t1_49983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_497202#
!psiPhi_t1/StatefulPartitionedCall�
!psiPhi_t2/StatefulPartitionedCallStatefulPartitionedCall base_t2/PartitionedCall:output:0psiphi_t2_49986psiphi_t2_49988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_497362#
!psiPhi_t2/StatefulPartitionedCall�
psi_layer/PartitionedCallPartitionedCall*psiPhi_t0/StatefulPartitionedCall:output:0*psiPhi_t1/StatefulPartitionedCall:output:0*psiPhi_t2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psi_layer_layer_call_and_return_conditional_losses_497502
psi_layer/PartitionedCall�
synthetic_trace/PartitionedCallPartitionedCallv"psi_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497582!
synthetic_trace/PartitionedCall�
IdentityIdentity(synthetic_trace/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^psiPhi_t0/StatefulPartitionedCall"^psiPhi_t1/StatefulPartitionedCall"^psiPhi_t2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2F
!psiPhi_t0/StatefulPartitionedCall!psiPhi_t0/StatefulPartitionedCall2F
!psiPhi_t1/StatefulPartitionedCall!psiPhi_t1/StatefulPartitionedCall2F
!psiPhi_t2/StatefulPartitionedCall!psiPhi_t2/StatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namev:_[
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis
�
^
B__inference_base_t1_layer_call_and_return_conditional_losses_49868

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_base_t1_layer_call_fn_50198

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_496822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
!__inference__traced_restore_50395
file_prefix4
!assignvariableop_psiphi_t0_kernel:	�/
!assignvariableop_1_psiphi_t0_bias:6
#assignvariableop_2_psiphi_t1_kernel:	�/
!assignvariableop_3_psiphi_t1_bias:6
#assignvariableop_4_psiphi_t2_kernel:	�/
!assignvariableop_5_psiphi_t2_bias:

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_psiphi_t0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_psiphi_t0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_psiphi_t1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_psiphi_t1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_psiphi_t2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_psiphi_t2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6c

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_7�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
C
'__inference_base_t1_layer_call_fn_50203

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_498682
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
D__inference_psi_layer_layer_call_and_return_conditional_losses_50294
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
^
B__inference_base_t1_layer_call_and_return_conditional_losses_50185

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
v
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50307
inputs_0
inputs_1
identityY
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
t
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_49792

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_49736

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_base_t2_layer_call_and_return_conditional_losses_49887

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_psiPhi_t1_layer_call_fn_50267

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_497202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_49720

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_decoder_layer_call_fn_49969
v
orthonormal_basis
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvorthonormal_basisunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_499362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namev:_[
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis
�

�
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_50239

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_base_t0_layer_call_fn_50177

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_498492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_base_t1_layer_call_and_return_conditional_losses_49682

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_50277

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
'__inference_decoder_layer_call_fn_50151
inputs_0
inputs_1
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_499362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
|
D__inference_psi_layer_layer_call_and_return_conditional_losses_49750

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
'__inference_decoder_layer_call_fn_49776
v
orthonormal_basis
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvorthonormal_basisunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_497612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namev:_[
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis
�

�
#__inference_signature_wrapper_50039
orthonormal_basis
v
unknown:	�
	unknown_0:
	unknown_1:	�
	unknown_2:
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvorthonormal_basisunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_496552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:����������:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis:JF
'
_output_shapes
:���������

_user_specified_namev
�
t
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_49758

inputs
inputs_1
identityW
addAddV2inputsinputs_1*
T0*'
_output_shapes
:���������2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_base_t0_layer_call_and_return_conditional_losses_49849

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
B__inference_decoder_layer_call_and_return_conditional_losses_49936

inputs
inputs_1"
psiphi_t0_49918:	�
psiphi_t0_49920:"
psiphi_t1_49923:	�
psiphi_t1_49925:"
psiphi_t2_49928:	�
psiphi_t2_49930:
identity��!psiPhi_t0/StatefulPartitionedCall�!psiPhi_t1/StatefulPartitionedCall�!psiPhi_t2/StatefulPartitionedCall�
base_t2/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_498872
base_t2/PartitionedCall�
base_t1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_498682
base_t1/PartitionedCall�
base_t0/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_498492
base_t0/PartitionedCall�
!psiPhi_t0/StatefulPartitionedCallStatefulPartitionedCall base_t0/PartitionedCall:output:0psiphi_t0_49918psiphi_t0_49920*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_497042#
!psiPhi_t0/StatefulPartitionedCall�
!psiPhi_t1/StatefulPartitionedCallStatefulPartitionedCall base_t1/PartitionedCall:output:0psiphi_t1_49923psiphi_t1_49925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_497202#
!psiPhi_t1/StatefulPartitionedCall�
!psiPhi_t2/StatefulPartitionedCallStatefulPartitionedCall base_t2/PartitionedCall:output:0psiphi_t2_49928psiphi_t2_49930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_497362#
!psiPhi_t2/StatefulPartitionedCall�
psi_layer/PartitionedCallPartitionedCall*psiPhi_t0/StatefulPartitionedCall:output:0*psiPhi_t1/StatefulPartitionedCall:output:0*psiPhi_t2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psi_layer_layer_call_and_return_conditional_losses_497502
psi_layer/PartitionedCall�
synthetic_trace/PartitionedCallPartitionedCallinputs"psi_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497922!
synthetic_trace/PartitionedCall�
IdentityIdentity(synthetic_trace/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^psiPhi_t0/StatefulPartitionedCall"^psiPhi_t1/StatefulPartitionedCall"^psiPhi_t2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2F
!psiPhi_t0/StatefulPartitionedCall!psiPhi_t0/StatefulPartitionedCall2F
!psiPhi_t1/StatefulPartitionedCall!psiPhi_t1/StatefulPartitionedCall2F
!psiPhi_t2/StatefulPartitionedCall!psiPhi_t2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_base_t0_layer_call_and_return_conditional_losses_50167

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�;
�
 __inference__wrapped_model_49655
v
orthonormal_basisC
0decoder_psiphi_t0_matmul_readvariableop_resource:	�?
1decoder_psiphi_t0_biasadd_readvariableop_resource:C
0decoder_psiphi_t1_matmul_readvariableop_resource:	�?
1decoder_psiphi_t1_biasadd_readvariableop_resource:C
0decoder_psiphi_t2_matmul_readvariableop_resource:	�?
1decoder_psiphi_t2_biasadd_readvariableop_resource:
identity��(decoder/psiPhi_t0/BiasAdd/ReadVariableOp�'decoder/psiPhi_t0/MatMul/ReadVariableOp�(decoder/psiPhi_t1/BiasAdd/ReadVariableOp�'decoder/psiPhi_t1/MatMul/ReadVariableOp�(decoder/psiPhi_t2/BiasAdd/ReadVariableOp�'decoder/psiPhi_t2/MatMul/ReadVariableOp�
#decoder/base_t2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2%
#decoder/base_t2/strided_slice/stack�
%decoder/base_t2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2'
%decoder/base_t2/strided_slice/stack_1�
%decoder/base_t2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2'
%decoder/base_t2/strided_slice/stack_2�
decoder/base_t2/strided_sliceStridedSliceorthonormal_basis,decoder/base_t2/strided_slice/stack:output:0.decoder/base_t2/strided_slice/stack_1:output:0.decoder/base_t2/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
decoder/base_t2/strided_slice�
#decoder/base_t1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2%
#decoder/base_t1/strided_slice/stack�
%decoder/base_t1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2'
%decoder/base_t1/strided_slice/stack_1�
%decoder/base_t1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2'
%decoder/base_t1/strided_slice/stack_2�
decoder/base_t1/strided_sliceStridedSliceorthonormal_basis,decoder/base_t1/strided_slice/stack:output:0.decoder/base_t1/strided_slice/stack_1:output:0.decoder/base_t1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
decoder/base_t1/strided_slice�
#decoder/base_t0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2%
#decoder/base_t0/strided_slice/stack�
%decoder/base_t0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2'
%decoder/base_t0/strided_slice/stack_1�
%decoder/base_t0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2'
%decoder/base_t0/strided_slice/stack_2�
decoder/base_t0/strided_sliceStridedSliceorthonormal_basis,decoder/base_t0/strided_slice/stack:output:0.decoder/base_t0/strided_slice/stack_1:output:0.decoder/base_t0/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
decoder/base_t0/strided_slice�
'decoder/psiPhi_t0/MatMul/ReadVariableOpReadVariableOp0decoder_psiphi_t0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'decoder/psiPhi_t0/MatMul/ReadVariableOp�
decoder/psiPhi_t0/MatMulMatMul&decoder/base_t0/strided_slice:output:0/decoder/psiPhi_t0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t0/MatMul�
(decoder/psiPhi_t0/BiasAdd/ReadVariableOpReadVariableOp1decoder_psiphi_t0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(decoder/psiPhi_t0/BiasAdd/ReadVariableOp�
decoder/psiPhi_t0/BiasAddBiasAdd"decoder/psiPhi_t0/MatMul:product:00decoder/psiPhi_t0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t0/BiasAdd�
'decoder/psiPhi_t1/MatMul/ReadVariableOpReadVariableOp0decoder_psiphi_t1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'decoder/psiPhi_t1/MatMul/ReadVariableOp�
decoder/psiPhi_t1/MatMulMatMul&decoder/base_t1/strided_slice:output:0/decoder/psiPhi_t1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t1/MatMul�
(decoder/psiPhi_t1/BiasAdd/ReadVariableOpReadVariableOp1decoder_psiphi_t1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(decoder/psiPhi_t1/BiasAdd/ReadVariableOp�
decoder/psiPhi_t1/BiasAddBiasAdd"decoder/psiPhi_t1/MatMul:product:00decoder/psiPhi_t1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t1/BiasAdd�
'decoder/psiPhi_t2/MatMul/ReadVariableOpReadVariableOp0decoder_psiphi_t2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'decoder/psiPhi_t2/MatMul/ReadVariableOp�
decoder/psiPhi_t2/MatMulMatMul&decoder/base_t2/strided_slice:output:0/decoder/psiPhi_t2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t2/MatMul�
(decoder/psiPhi_t2/BiasAdd/ReadVariableOpReadVariableOp1decoder_psiphi_t2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(decoder/psiPhi_t2/BiasAdd/ReadVariableOp�
decoder/psiPhi_t2/BiasAddBiasAdd"decoder/psiPhi_t2/MatMul:product:00decoder/psiPhi_t2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/psiPhi_t2/BiasAdd�
decoder/psi_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
decoder/psi_layer/concat/axis�
decoder/psi_layer/concatConcatV2"decoder/psiPhi_t0/BiasAdd:output:0"decoder/psiPhi_t1/BiasAdd:output:0"decoder/psiPhi_t2/BiasAdd:output:0&decoder/psi_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
decoder/psi_layer/concat�
decoder/synthetic_trace/addAddV2v!decoder/psi_layer/concat:output:0*
T0*'
_output_shapes
:���������2
decoder/synthetic_trace/addz
IdentityIdentitydecoder/synthetic_trace/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^decoder/psiPhi_t0/BiasAdd/ReadVariableOp(^decoder/psiPhi_t0/MatMul/ReadVariableOp)^decoder/psiPhi_t1/BiasAdd/ReadVariableOp(^decoder/psiPhi_t1/MatMul/ReadVariableOp)^decoder/psiPhi_t2/BiasAdd/ReadVariableOp(^decoder/psiPhi_t2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2T
(decoder/psiPhi_t0/BiasAdd/ReadVariableOp(decoder/psiPhi_t0/BiasAdd/ReadVariableOp2R
'decoder/psiPhi_t0/MatMul/ReadVariableOp'decoder/psiPhi_t0/MatMul/ReadVariableOp2T
(decoder/psiPhi_t1/BiasAdd/ReadVariableOp(decoder/psiPhi_t1/BiasAdd/ReadVariableOp2R
'decoder/psiPhi_t1/MatMul/ReadVariableOp'decoder/psiPhi_t1/MatMul/ReadVariableOp2T
(decoder/psiPhi_t2/BiasAdd/ReadVariableOp(decoder/psiPhi_t2/BiasAdd/ReadVariableOp2R
'decoder/psiPhi_t2/MatMul/ReadVariableOp'decoder/psiPhi_t2/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������

_user_specified_namev:_[
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis
�
^
B__inference_base_t2_layer_call_and_return_conditional_losses_49672

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
C
'__inference_base_t0_layer_call_fn_50172

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_496922
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_psiPhi_t0_layer_call_fn_50248

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_497042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
[
/__inference_synthetic_trace_layer_call_fn_50325
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�3
�
B__inference_decoder_layer_call_and_return_conditional_losses_50077
inputs_0
inputs_1;
(psiphi_t0_matmul_readvariableop_resource:	�7
)psiphi_t0_biasadd_readvariableop_resource:;
(psiphi_t1_matmul_readvariableop_resource:	�7
)psiphi_t1_biasadd_readvariableop_resource:;
(psiphi_t2_matmul_readvariableop_resource:	�7
)psiphi_t2_biasadd_readvariableop_resource:
identity�� psiPhi_t0/BiasAdd/ReadVariableOp�psiPhi_t0/MatMul/ReadVariableOp� psiPhi_t1/BiasAdd/ReadVariableOp�psiPhi_t1/MatMul/ReadVariableOp� psiPhi_t2/BiasAdd/ReadVariableOp�psiPhi_t2/MatMul/ReadVariableOp�
base_t2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t2/strided_slice/stack�
base_t2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t2/strided_slice/stack_1�
base_t2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t2/strided_slice/stack_2�
base_t2/strided_sliceStridedSliceinputs_1$base_t2/strided_slice/stack:output:0&base_t2/strided_slice/stack_1:output:0&base_t2/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t2/strided_slice�
base_t1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t1/strided_slice/stack�
base_t1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t1/strided_slice/stack_1�
base_t1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t1/strided_slice/stack_2�
base_t1/strided_sliceStridedSliceinputs_1$base_t1/strided_slice/stack:output:0&base_t1/strided_slice/stack_1:output:0&base_t1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t1/strided_slice�
base_t0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t0/strided_slice/stack�
base_t0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t0/strided_slice/stack_1�
base_t0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t0/strided_slice/stack_2�
base_t0/strided_sliceStridedSliceinputs_1$base_t0/strided_slice/stack:output:0&base_t0/strided_slice/stack_1:output:0&base_t0/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t0/strided_slice�
psiPhi_t0/MatMul/ReadVariableOpReadVariableOp(psiphi_t0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t0/MatMul/ReadVariableOp�
psiPhi_t0/MatMulMatMulbase_t0/strided_slice:output:0'psiPhi_t0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t0/MatMul�
 psiPhi_t0/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t0/BiasAdd/ReadVariableOp�
psiPhi_t0/BiasAddBiasAddpsiPhi_t0/MatMul:product:0(psiPhi_t0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t0/BiasAdd�
psiPhi_t1/MatMul/ReadVariableOpReadVariableOp(psiphi_t1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t1/MatMul/ReadVariableOp�
psiPhi_t1/MatMulMatMulbase_t1/strided_slice:output:0'psiPhi_t1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t1/MatMul�
 psiPhi_t1/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t1/BiasAdd/ReadVariableOp�
psiPhi_t1/BiasAddBiasAddpsiPhi_t1/MatMul:product:0(psiPhi_t1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t1/BiasAdd�
psiPhi_t2/MatMul/ReadVariableOpReadVariableOp(psiphi_t2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t2/MatMul/ReadVariableOp�
psiPhi_t2/MatMulMatMulbase_t2/strided_slice:output:0'psiPhi_t2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t2/MatMul�
 psiPhi_t2/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t2/BiasAdd/ReadVariableOp�
psiPhi_t2/BiasAddBiasAddpsiPhi_t2/MatMul:product:0(psiPhi_t2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t2/BiasAddp
psi_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
psi_layer/concat/axis�
psi_layer/concatConcatV2psiPhi_t0/BiasAdd:output:0psiPhi_t1/BiasAdd:output:0psiPhi_t2/BiasAdd:output:0psi_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
psi_layer/concat�
synthetic_trace/addAddV2inputs_0psi_layer/concat:output:0*
T0*'
_output_shapes
:���������2
synthetic_trace/addr
IdentityIdentitysynthetic_trace/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^psiPhi_t0/BiasAdd/ReadVariableOp ^psiPhi_t0/MatMul/ReadVariableOp!^psiPhi_t1/BiasAdd/ReadVariableOp ^psiPhi_t1/MatMul/ReadVariableOp!^psiPhi_t2/BiasAdd/ReadVariableOp ^psiPhi_t2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2D
 psiPhi_t0/BiasAdd/ReadVariableOp psiPhi_t0/BiasAdd/ReadVariableOp2B
psiPhi_t0/MatMul/ReadVariableOppsiPhi_t0/MatMul/ReadVariableOp2D
 psiPhi_t1/BiasAdd/ReadVariableOp psiPhi_t1/BiasAdd/ReadVariableOp2B
psiPhi_t1/MatMul/ReadVariableOppsiPhi_t1/MatMul/ReadVariableOp2D
 psiPhi_t2/BiasAdd/ReadVariableOp psiPhi_t2/BiasAdd/ReadVariableOp2B
psiPhi_t2/MatMul/ReadVariableOppsiPhi_t2/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
^
B__inference_base_t1_layer_call_and_return_conditional_losses_50193

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack�
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1�
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2�
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slicek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
[
/__inference_synthetic_trace_layer_call_fn_50319
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�#
�
B__inference_decoder_layer_call_and_return_conditional_losses_50019
v
orthonormal_basis"
psiphi_t0_50001:	�
psiphi_t0_50003:"
psiphi_t1_50006:	�
psiphi_t1_50008:"
psiphi_t2_50011:	�
psiphi_t2_50013:
identity��!psiPhi_t0/StatefulPartitionedCall�!psiPhi_t1/StatefulPartitionedCall�!psiPhi_t2/StatefulPartitionedCall�
base_t2/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_498872
base_t2/PartitionedCall�
base_t1/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_498682
base_t1/PartitionedCall�
base_t0/PartitionedCallPartitionedCallorthonormal_basis*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_498492
base_t0/PartitionedCall�
!psiPhi_t0/StatefulPartitionedCallStatefulPartitionedCall base_t0/PartitionedCall:output:0psiphi_t0_50001psiphi_t0_50003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_497042#
!psiPhi_t0/StatefulPartitionedCall�
!psiPhi_t1/StatefulPartitionedCallStatefulPartitionedCall base_t1/PartitionedCall:output:0psiphi_t1_50006psiphi_t1_50008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_497202#
!psiPhi_t1/StatefulPartitionedCall�
!psiPhi_t2/StatefulPartitionedCallStatefulPartitionedCall base_t2/PartitionedCall:output:0psiphi_t2_50011psiphi_t2_50013*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_497362#
!psiPhi_t2/StatefulPartitionedCall�
psi_layer/PartitionedCallPartitionedCall*psiPhi_t0/StatefulPartitionedCall:output:0*psiPhi_t1/StatefulPartitionedCall:output:0*psiPhi_t2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psi_layer_layer_call_and_return_conditional_losses_497502
psi_layer/PartitionedCall�
synthetic_trace/PartitionedCallPartitionedCallv"psi_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497922!
synthetic_trace/PartitionedCall�
IdentityIdentity(synthetic_trace/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^psiPhi_t0/StatefulPartitionedCall"^psiPhi_t1/StatefulPartitionedCall"^psiPhi_t2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2F
!psiPhi_t0/StatefulPartitionedCall!psiPhi_t0/StatefulPartitionedCall2F
!psiPhi_t1/StatefulPartitionedCall!psiPhi_t1/StatefulPartitionedCall2F
!psiPhi_t2/StatefulPartitionedCall!psiPhi_t2/StatefulPartitionedCall:J F
'
_output_shapes
:���������

_user_specified_namev:_[
,
_output_shapes
:����������
+
_user_specified_nameorthonormal_basis
�#
�
B__inference_decoder_layer_call_and_return_conditional_losses_49761

inputs
inputs_1"
psiphi_t0_49705:	�
psiphi_t0_49707:"
psiphi_t1_49721:	�
psiphi_t1_49723:"
psiphi_t2_49737:	�
psiphi_t2_49739:
identity��!psiPhi_t0/StatefulPartitionedCall�!psiPhi_t1/StatefulPartitionedCall�!psiPhi_t2/StatefulPartitionedCall�
base_t2/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_496722
base_t2/PartitionedCall�
base_t1/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t1_layer_call_and_return_conditional_losses_496822
base_t1/PartitionedCall�
base_t0/PartitionedCallPartitionedCallinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t0_layer_call_and_return_conditional_losses_496922
base_t0/PartitionedCall�
!psiPhi_t0/StatefulPartitionedCallStatefulPartitionedCall base_t0/PartitionedCall:output:0psiphi_t0_49705psiphi_t0_49707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_497042#
!psiPhi_t0/StatefulPartitionedCall�
!psiPhi_t1/StatefulPartitionedCallStatefulPartitionedCall base_t1/PartitionedCall:output:0psiphi_t1_49721psiphi_t1_49723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_497202#
!psiPhi_t1/StatefulPartitionedCall�
!psiPhi_t2/StatefulPartitionedCallStatefulPartitionedCall base_t2/PartitionedCall:output:0psiphi_t2_49737psiphi_t2_49739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_497362#
!psiPhi_t2/StatefulPartitionedCall�
psi_layer/PartitionedCallPartitionedCall*psiPhi_t0/StatefulPartitionedCall:output:0*psiPhi_t1/StatefulPartitionedCall:output:0*psiPhi_t2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psi_layer_layer_call_and_return_conditional_losses_497502
psi_layer/PartitionedCall�
synthetic_trace/PartitionedCallPartitionedCallinputs"psi_layer/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_497582!
synthetic_trace/PartitionedCall�
IdentityIdentity(synthetic_trace/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^psiPhi_t0/StatefulPartitionedCall"^psiPhi_t1/StatefulPartitionedCall"^psiPhi_t2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2F
!psiPhi_t0/StatefulPartitionedCall!psiPhi_t0/StatefulPartitionedCall2F
!psiPhi_t1/StatefulPartitionedCall!psiPhi_t1/StatefulPartitionedCall2F
!psiPhi_t2/StatefulPartitionedCall!psiPhi_t2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�3
�
B__inference_decoder_layer_call_and_return_conditional_losses_50115
inputs_0
inputs_1;
(psiphi_t0_matmul_readvariableop_resource:	�7
)psiphi_t0_biasadd_readvariableop_resource:;
(psiphi_t1_matmul_readvariableop_resource:	�7
)psiphi_t1_biasadd_readvariableop_resource:;
(psiphi_t2_matmul_readvariableop_resource:	�7
)psiphi_t2_biasadd_readvariableop_resource:
identity�� psiPhi_t0/BiasAdd/ReadVariableOp�psiPhi_t0/MatMul/ReadVariableOp� psiPhi_t1/BiasAdd/ReadVariableOp�psiPhi_t1/MatMul/ReadVariableOp� psiPhi_t2/BiasAdd/ReadVariableOp�psiPhi_t2/MatMul/ReadVariableOp�
base_t2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t2/strided_slice/stack�
base_t2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t2/strided_slice/stack_1�
base_t2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t2/strided_slice/stack_2�
base_t2/strided_sliceStridedSliceinputs_1$base_t2/strided_slice/stack:output:0&base_t2/strided_slice/stack_1:output:0&base_t2/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t2/strided_slice�
base_t1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t1/strided_slice/stack�
base_t1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t1/strided_slice/stack_1�
base_t1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t1/strided_slice/stack_2�
base_t1/strided_sliceStridedSliceinputs_1$base_t1/strided_slice/stack:output:0&base_t1/strided_slice/stack_1:output:0&base_t1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t1/strided_slice�
base_t0/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
base_t0/strided_slice/stack�
base_t0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
base_t0/strided_slice/stack_1�
base_t0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
base_t0/strided_slice/stack_2�
base_t0/strided_sliceStridedSliceinputs_1$base_t0/strided_slice/stack:output:0&base_t0/strided_slice/stack_1:output:0&base_t0/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*

begin_mask*
end_mask*
shrink_axis_mask2
base_t0/strided_slice�
psiPhi_t0/MatMul/ReadVariableOpReadVariableOp(psiphi_t0_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t0/MatMul/ReadVariableOp�
psiPhi_t0/MatMulMatMulbase_t0/strided_slice:output:0'psiPhi_t0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t0/MatMul�
 psiPhi_t0/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t0/BiasAdd/ReadVariableOp�
psiPhi_t0/BiasAddBiasAddpsiPhi_t0/MatMul:product:0(psiPhi_t0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t0/BiasAdd�
psiPhi_t1/MatMul/ReadVariableOpReadVariableOp(psiphi_t1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t1/MatMul/ReadVariableOp�
psiPhi_t1/MatMulMatMulbase_t1/strided_slice:output:0'psiPhi_t1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t1/MatMul�
 psiPhi_t1/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t1/BiasAdd/ReadVariableOp�
psiPhi_t1/BiasAddBiasAddpsiPhi_t1/MatMul:product:0(psiPhi_t1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t1/BiasAdd�
psiPhi_t2/MatMul/ReadVariableOpReadVariableOp(psiphi_t2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02!
psiPhi_t2/MatMul/ReadVariableOp�
psiPhi_t2/MatMulMatMulbase_t2/strided_slice:output:0'psiPhi_t2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t2/MatMul�
 psiPhi_t2/BiasAdd/ReadVariableOpReadVariableOp)psiphi_t2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 psiPhi_t2/BiasAdd/ReadVariableOp�
psiPhi_t2/BiasAddBiasAddpsiPhi_t2/MatMul:product:0(psiPhi_t2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
psiPhi_t2/BiasAddp
psi_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
psi_layer/concat/axis�
psi_layer/concatConcatV2psiPhi_t0/BiasAdd:output:0psiPhi_t1/BiasAdd:output:0psiPhi_t2/BiasAdd:output:0psi_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
psi_layer/concat�
synthetic_trace/addAddV2inputs_0psi_layer/concat:output:0*
T0*'
_output_shapes
:���������2
synthetic_trace/addr
IdentityIdentitysynthetic_trace/add:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^psiPhi_t0/BiasAdd/ReadVariableOp ^psiPhi_t0/MatMul/ReadVariableOp!^psiPhi_t1/BiasAdd/ReadVariableOp ^psiPhi_t1/MatMul/ReadVariableOp!^psiPhi_t2/BiasAdd/ReadVariableOp ^psiPhi_t2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������:����������: : : : : : 2D
 psiPhi_t0/BiasAdd/ReadVariableOp psiPhi_t0/BiasAdd/ReadVariableOp2B
psiPhi_t0/MatMul/ReadVariableOppsiPhi_t0/MatMul/ReadVariableOp2D
 psiPhi_t1/BiasAdd/ReadVariableOp psiPhi_t1/BiasAdd/ReadVariableOp2B
psiPhi_t1/MatMul/ReadVariableOppsiPhi_t1/MatMul/ReadVariableOp2D
 psiPhi_t2/BiasAdd/ReadVariableOp psiPhi_t2/BiasAdd/ReadVariableOp2B
psiPhi_t2/MatMul/ReadVariableOppsiPhi_t2/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
c
)__inference_psi_layer_layer_call_fn_50301
inputs_0
inputs_1
inputs_2
identity�
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_psi_layer_layer_call_and_return_conditional_losses_497502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:���������:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/2
�
C
'__inference_base_t2_layer_call_fn_50224

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_base_t2_layer_call_and_return_conditional_losses_496722
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
T
orthonormal_basis?
#serving_default_orthonormal_basis:0����������
/
v*
serving_default_v:0���������C
synthetic_trace0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
regularization_losses
	variables
trainable_variables
	keras_api

signatures
c_default_save_signature
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_network
"
_tf_keras_input_layer
�
regularization_losses
	variables
trainable_variables
	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
�
regularization_losses
	variables
trainable_variables
	keras_api
*h&call_and_return_all_conditional_losses
i__call__"
_tf_keras_layer
�
regularization_losses
	variables
trainable_variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"
_tf_keras_layer
�

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_layer
�

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
*n&call_and_return_all_conditional_losses
o__call__"
_tf_keras_layer
�

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_layer
"
_tf_keras_input_layer
�
.regularization_losses
/	variables
0trainable_variables
1	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_layer
�
2regularization_losses
3	variables
4trainable_variables
5	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
 "
trackable_list_wrapper
J
0
1
"2
#3
(4
)5"
trackable_list_wrapper
J
0
1
"2
#3
(4
)5"
trackable_list_wrapper
�
regularization_losses

6layers
7layer_metrics
	variables
8layer_regularization_losses
trainable_variables
9non_trainable_variables
:metrics
e__call__
c_default_save_signature
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses

;layers
<layer_metrics
	variables
=layer_regularization_losses
trainable_variables
>non_trainable_variables
?metrics
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses

@layers
Alayer_metrics
	variables
Blayer_regularization_losses
trainable_variables
Cnon_trainable_variables
Dmetrics
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses

Elayers
Flayer_metrics
	variables
Glayer_regularization_losses
trainable_variables
Hnon_trainable_variables
Imetrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
#:!	�2psiPhi_t0/kernel
:2psiPhi_t0/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses

Jlayers
Klayer_metrics
	variables
Llayer_regularization_losses
 trainable_variables
Mnon_trainable_variables
Nmetrics
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
#:!	�2psiPhi_t1/kernel
:2psiPhi_t1/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
$regularization_losses

Olayers
Player_metrics
%	variables
Qlayer_regularization_losses
&trainable_variables
Rnon_trainable_variables
Smetrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
#:!	�2psiPhi_t2/kernel
:2psiPhi_t2/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
*regularization_losses

Tlayers
Ulayer_metrics
+	variables
Vlayer_regularization_losses
,trainable_variables
Wnon_trainable_variables
Xmetrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
.regularization_losses

Ylayers
Zlayer_metrics
/	variables
[layer_regularization_losses
0trainable_variables
\non_trainable_variables
]metrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
2regularization_losses

^layers
_layer_metrics
3	variables
`layer_regularization_losses
4trainable_variables
anon_trainable_variables
bmetrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
 __inference__wrapped_model_49655vorthonormal_basis"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_decoder_layer_call_and_return_conditional_losses_50077
B__inference_decoder_layer_call_and_return_conditional_losses_50115
B__inference_decoder_layer_call_and_return_conditional_losses_49994
B__inference_decoder_layer_call_and_return_conditional_losses_50019�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_decoder_layer_call_fn_49776
'__inference_decoder_layer_call_fn_50133
'__inference_decoder_layer_call_fn_50151
'__inference_decoder_layer_call_fn_49969�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_base_t0_layer_call_and_return_conditional_losses_50159
B__inference_base_t0_layer_call_and_return_conditional_losses_50167�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_base_t0_layer_call_fn_50172
'__inference_base_t0_layer_call_fn_50177�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_base_t1_layer_call_and_return_conditional_losses_50185
B__inference_base_t1_layer_call_and_return_conditional_losses_50193�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_base_t1_layer_call_fn_50198
'__inference_base_t1_layer_call_fn_50203�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_base_t2_layer_call_and_return_conditional_losses_50211
B__inference_base_t2_layer_call_and_return_conditional_losses_50219�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_base_t2_layer_call_fn_50224
'__inference_base_t2_layer_call_fn_50229�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_50239�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_psiPhi_t0_layer_call_fn_50248�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_50258�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_psiPhi_t1_layer_call_fn_50267�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_50277�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_psiPhi_t2_layer_call_fn_50286�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_psi_layer_layer_call_and_return_conditional_losses_50294�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_psi_layer_layer_call_fn_50301�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50307
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50313�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
/__inference_synthetic_trace_layer_call_fn_50319
/__inference_synthetic_trace_layer_call_fn_50325�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
#__inference_signature_wrapper_50039orthonormal_basisv"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_49655�"#()a�^
W�T
R�O
�
v���������
0�-
orthonormal_basis����������
� "A�>
<
synthetic_trace)�&
synthetic_trace����������
B__inference_base_t0_layer_call_and_return_conditional_losses_50159f<�9
2�/
%�"
inputs����������

 
p 
� "&�#
�
0����������
� �
B__inference_base_t0_layer_call_and_return_conditional_losses_50167f<�9
2�/
%�"
inputs����������

 
p
� "&�#
�
0����������
� �
'__inference_base_t0_layer_call_fn_50172Y<�9
2�/
%�"
inputs����������

 
p 
� "������������
'__inference_base_t0_layer_call_fn_50177Y<�9
2�/
%�"
inputs����������

 
p
� "������������
B__inference_base_t1_layer_call_and_return_conditional_losses_50185f<�9
2�/
%�"
inputs����������

 
p 
� "&�#
�
0����������
� �
B__inference_base_t1_layer_call_and_return_conditional_losses_50193f<�9
2�/
%�"
inputs����������

 
p
� "&�#
�
0����������
� �
'__inference_base_t1_layer_call_fn_50198Y<�9
2�/
%�"
inputs����������

 
p 
� "������������
'__inference_base_t1_layer_call_fn_50203Y<�9
2�/
%�"
inputs����������

 
p
� "������������
B__inference_base_t2_layer_call_and_return_conditional_losses_50211f<�9
2�/
%�"
inputs����������

 
p 
� "&�#
�
0����������
� �
B__inference_base_t2_layer_call_and_return_conditional_losses_50219f<�9
2�/
%�"
inputs����������

 
p
� "&�#
�
0����������
� �
'__inference_base_t2_layer_call_fn_50224Y<�9
2�/
%�"
inputs����������

 
p 
� "������������
'__inference_base_t2_layer_call_fn_50229Y<�9
2�/
%�"
inputs����������

 
p
� "������������
B__inference_decoder_layer_call_and_return_conditional_losses_49994�"#()i�f
_�\
R�O
�
v���������
0�-
orthonormal_basis����������
p 

 
� "%�"
�
0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_50019�"#()i�f
_�\
R�O
�
v���������
0�-
orthonormal_basis����������
p

 
� "%�"
�
0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_50077�"#()g�d
]�Z
P�M
"�
inputs/0���������
'�$
inputs/1����������
p 

 
� "%�"
�
0���������
� �
B__inference_decoder_layer_call_and_return_conditional_losses_50115�"#()g�d
]�Z
P�M
"�
inputs/0���������
'�$
inputs/1����������
p

 
� "%�"
�
0���������
� �
'__inference_decoder_layer_call_fn_49776�"#()i�f
_�\
R�O
�
v���������
0�-
orthonormal_basis����������
p 

 
� "�����������
'__inference_decoder_layer_call_fn_49969�"#()i�f
_�\
R�O
�
v���������
0�-
orthonormal_basis����������
p

 
� "�����������
'__inference_decoder_layer_call_fn_50133�"#()g�d
]�Z
P�M
"�
inputs/0���������
'�$
inputs/1����������
p 

 
� "�����������
'__inference_decoder_layer_call_fn_50151�"#()g�d
]�Z
P�M
"�
inputs/0���������
'�$
inputs/1����������
p

 
� "�����������
D__inference_psiPhi_t0_layer_call_and_return_conditional_losses_50239]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_psiPhi_t0_layer_call_fn_50248P0�-
&�#
!�
inputs����������
� "�����������
D__inference_psiPhi_t1_layer_call_and_return_conditional_losses_50258]"#0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_psiPhi_t1_layer_call_fn_50267P"#0�-
&�#
!�
inputs����������
� "�����������
D__inference_psiPhi_t2_layer_call_and_return_conditional_losses_50277]()0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� }
)__inference_psiPhi_t2_layer_call_fn_50286P()0�-
&�#
!�
inputs����������
� "�����������
D__inference_psi_layer_layer_call_and_return_conditional_losses_50294�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "%�"
�
0���������
� �
)__inference_psi_layer_layer_call_fn_50301�~�{
t�q
o�l
"�
inputs/0���������
"�
inputs/1���������
"�
inputs/2���������
� "�����������
#__inference_signature_wrapper_50039�"#()v�s
� 
l�i
E
orthonormal_basis0�-
orthonormal_basis����������
 
v�
v���������"A�>
<
synthetic_trace)�&
synthetic_trace����������
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50307�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p 
� "%�"
�
0���������
� �
J__inference_synthetic_trace_layer_call_and_return_conditional_losses_50313�b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p
� "%�"
�
0���������
� �
/__inference_synthetic_trace_layer_call_fn_50319~b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p 
� "�����������
/__inference_synthetic_trace_layer_call_fn_50325~b�_
X�U
K�H
"�
inputs/0���������
"�
inputs/1���������

 
p
� "����������