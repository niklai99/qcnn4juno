??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8??
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
: *
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
: *
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
: @*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:@*
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
:@@*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:d*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:d
*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
SGD/conv_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv_1/kernel/momentum
?
.SGD/conv_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_1/kernel/momentum*&
_output_shapes
: *
dtype0
?
SGD/conv_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameSGD/conv_1/bias/momentum
?
,SGD/conv_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_1/bias/momentum*
_output_shapes
: *
dtype0
?
SGD/conv_2/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameSGD/conv_2/kernel/momentum
?
.SGD/conv_2/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_2/kernel/momentum*&
_output_shapes
: @*
dtype0
?
SGD/conv_2/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameSGD/conv_2/bias/momentum
?
,SGD/conv_2/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_2/bias/momentum*
_output_shapes
:@*
dtype0
?
SGD/conv_3/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_nameSGD/conv_3/kernel/momentum
?
.SGD/conv_3/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_3/kernel/momentum*&
_output_shapes
:@@*
dtype0
?
SGD/conv_3/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameSGD/conv_3/bias/momentum
?
,SGD/conv_3/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv_3/bias/momentum*
_output_shapes
:@*
dtype0
?
SGD/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?d*,
shared_nameSGD/dense_1/kernel/momentum
?
/SGD/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/kernel/momentum*
_output_shapes
:	?d*
dtype0
?
SGD/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_nameSGD/dense_1/bias/momentum
?
-SGD/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_1/bias/momentum*
_output_shapes
:d*
dtype0
?
SGD/output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d
*+
shared_nameSGD/output/kernel/momentum
?
.SGD/output/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/output/kernel/momentum*
_output_shapes

:d
*
dtype0
?
SGD/output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameSGD/output/bias/momentum
?
,SGD/output/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/output/bias/momentum*
_output_shapes
:
*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?/B?/ B?/
?

conv_1

pool_1

conv_2

conv_3

pool_2
flatten
dense_1
out
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
h

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?
9iter
	:decay
;learning_rate
<momentummomentumumomentumvmomentumwmomentumxmomentumy momentumz-momentum{.momentum|3momentum}4momentum~
F
0
1
2
3
4
 5
-6
.7
38
49
F
0
1
2
3
4
 5
-6
.7
38
49
 
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics

	variables
trainable_variables
regularization_losses
 
KI
VARIABLE_VALUEconv_1/kernel(conv_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_1/bias&conv_1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
KI
VARIABLE_VALUEconv_2/kernel(conv_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_2/bias&conv_2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
KI
VARIABLE_VALUEconv_3/kernel(conv_3/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_3/bias&conv_3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
!	variables
"trainable_variables
#regularization_losses
 
 
 
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
%	variables
&trainable_variables
'regularization_losses
 
 
 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
)	variables
*trainable_variables
+regularization_losses
MK
VARIABLE_VALUEdense_1/kernel)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_1/bias'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
/	variables
0trainable_variables
1regularization_losses
HF
VARIABLE_VALUEoutput/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEoutput/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41

30
41
 
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
5	variables
6trainable_variables
7regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

j0
k1
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
4
	ltotal
	mcount
n	variables
o	keras_api
D
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

n	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

p0
q1

s	variables
{y
VARIABLE_VALUESGD/conv_1/kernel/momentumKconv_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUESGD/conv_1/bias/momentumIconv_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUESGD/conv_2/kernel/momentumKconv_2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUESGD/conv_2/bias/momentumIconv_2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUESGD/conv_3/kernel/momentumKconv_3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUESGD/conv_3/bias/momentumIconv_3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUESGD/dense_1/kernel/momentumLdense_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUESGD/dense_1/bias/momentumJdense_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUESGD/output/kernel/momentumHout/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUESGD/output/bias/momentumFout/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasdense_1/kerneldense_1/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_302771
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.SGD/conv_1/kernel/momentum/Read/ReadVariableOp,SGD/conv_1/bias/momentum/Read/ReadVariableOp.SGD/conv_2/kernel/momentum/Read/ReadVariableOp,SGD/conv_2/bias/momentum/Read/ReadVariableOp.SGD/conv_3/kernel/momentum/Read/ReadVariableOp,SGD/conv_3/bias/momentum/Read/ReadVariableOp/SGD/dense_1/kernel/momentum/Read/ReadVariableOp-SGD/dense_1/bias/momentum/Read/ReadVariableOp.SGD/output/kernel/momentum/Read/ReadVariableOp,SGD/output/bias/momentum/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_303097
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasdense_1/kerneldense_1/biasoutput/kerneloutput/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1SGD/conv_1/kernel/momentumSGD/conv_1/bias/momentumSGD/conv_2/kernel/momentumSGD/conv_2/bias/momentumSGD/conv_3/kernel/momentumSGD/conv_3/bias/momentumSGD/dense_1/kernel/momentumSGD/dense_1/bias/momentumSGD/output/kernel/momentumSGD/output/bias/momentum*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_303191??
?1
?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302839

inputs?
%conv_1_conv2d_readvariableop_resource: 4
&conv_1_biasadd_readvariableop_resource: ?
%conv_2_conv2d_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@?
%conv_3_conv2d_readvariableop_resource:@@4
&conv_3_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	?d5
'dense_1_biasadd_readvariableop_resource:d7
%output_matmul_readvariableop_resource:d
4
&output_biasadd_readvariableop_resource:

identity??conv_1/BiasAdd/ReadVariableOp?conv_1/Conv2D/ReadVariableOp?conv_2/BiasAdd/ReadVariableOp?conv_2/Conv2D/ReadVariableOp?conv_3/BiasAdd/ReadVariableOp?conv_3/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
pool_1/MaxPoolMaxPoolconv_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv_2/Conv2DConv2Dpool_1/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@f
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
?
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@f
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@?
pool_2/MaxPoolMaxPoolconv_3/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ~
flatten/ReshapeReshapepool_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0?
output/MatMulMatMuldense_1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
d
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
g
IdentityIdentityoutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_pool_1_layer_call_fn_302869

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_302509h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
'__inference_conv_1_layer_call_fn_302848

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_302499w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_pool_1_layer_call_and_return_conditional_losses_302879

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_pool_2_layer_call_and_return_conditional_losses_302478

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_302970

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv_1_layer_call_and_return_conditional_losses_302859

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_pool_2_layer_call_fn_302929

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_302549h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		@:W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?
?
B__inference_conv_3_layer_call_and_return_conditional_losses_302919

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
^
B__inference_pool_2_layer_call_and_return_conditional_losses_302549

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		@:W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?
?
B__inference_conv_2_layer_call_and_return_conditional_losses_302899

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_pool_2_layer_call_and_return_conditional_losses_302934

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv_1_layer_call_and_return_conditional_losses_302499

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?6
?
!__inference__wrapped_model_302457
input_1E
+cnn_2_conv_1_conv2d_readvariableop_resource: :
,cnn_2_conv_1_biasadd_readvariableop_resource: E
+cnn_2_conv_2_conv2d_readvariableop_resource: @:
,cnn_2_conv_2_biasadd_readvariableop_resource:@E
+cnn_2_conv_3_conv2d_readvariableop_resource:@@:
,cnn_2_conv_3_biasadd_readvariableop_resource:@?
,cnn_2_dense_1_matmul_readvariableop_resource:	?d;
-cnn_2_dense_1_biasadd_readvariableop_resource:d=
+cnn_2_output_matmul_readvariableop_resource:d
:
,cnn_2_output_biasadd_readvariableop_resource:

identity??#cnn_2/conv_1/BiasAdd/ReadVariableOp?"cnn_2/conv_1/Conv2D/ReadVariableOp?#cnn_2/conv_2/BiasAdd/ReadVariableOp?"cnn_2/conv_2/Conv2D/ReadVariableOp?#cnn_2/conv_3/BiasAdd/ReadVariableOp?"cnn_2/conv_3/Conv2D/ReadVariableOp?$cnn_2/dense_1/BiasAdd/ReadVariableOp?#cnn_2/dense_1/MatMul/ReadVariableOp?#cnn_2/output/BiasAdd/ReadVariableOp?"cnn_2/output/MatMul/ReadVariableOp?
"cnn_2/conv_1/Conv2D/ReadVariableOpReadVariableOp+cnn_2_conv_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
cnn_2/conv_1/Conv2DConv2Dinput_1*cnn_2/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
#cnn_2/conv_1/BiasAdd/ReadVariableOpReadVariableOp,cnn_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_2/conv_1/BiasAddBiasAddcnn_2/conv_1/Conv2D:output:0+cnn_2/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? r
cnn_2/conv_1/ReluRelucnn_2/conv_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
cnn_2/pool_1/MaxPoolMaxPoolcnn_2/conv_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
"cnn_2/conv_2/Conv2D/ReadVariableOpReadVariableOp+cnn_2_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
cnn_2/conv_2/Conv2DConv2Dcnn_2/pool_1/MaxPool:output:0*cnn_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
#cnn_2/conv_2/BiasAdd/ReadVariableOpReadVariableOp,cnn_2_conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_2/conv_2/BiasAddBiasAddcnn_2/conv_2/Conv2D:output:0+cnn_2/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
cnn_2/conv_2/ReluRelucnn_2/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
"cnn_2/conv_3/Conv2D/ReadVariableOpReadVariableOp+cnn_2_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
cnn_2/conv_3/Conv2DConv2Dcnn_2/conv_2/Relu:activations:0*cnn_2/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
?
#cnn_2/conv_3/BiasAdd/ReadVariableOpReadVariableOp,cnn_2_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
cnn_2/conv_3/BiasAddBiasAddcnn_2/conv_3/Conv2D:output:0+cnn_2/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@r
cnn_2/conv_3/ReluRelucnn_2/conv_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@?
cnn_2/pool_2/MaxPoolMaxPoolcnn_2/conv_3/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
d
cnn_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
cnn_2/flatten/ReshapeReshapecnn_2/pool_2/MaxPool:output:0cnn_2/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
#cnn_2/dense_1/MatMul/ReadVariableOpReadVariableOp,cnn_2_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0?
cnn_2/dense_1/MatMulMatMulcnn_2/flatten/Reshape:output:0+cnn_2/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$cnn_2/dense_1/BiasAdd/ReadVariableOpReadVariableOp-cnn_2_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
cnn_2/dense_1/BiasAddBiasAddcnn_2/dense_1/MatMul:product:0,cnn_2/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dl
cnn_2/dense_1/ReluRelucnn_2/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d?
"cnn_2/output/MatMul/ReadVariableOpReadVariableOp+cnn_2_output_matmul_readvariableop_resource*
_output_shapes

:d
*
dtype0?
cnn_2/output/MatMulMatMul cnn_2/dense_1/Relu:activations:0*cnn_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
#cnn_2/output/BiasAdd/ReadVariableOpReadVariableOp,cnn_2_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
cnn_2/output/BiasAddBiasAddcnn_2/output/MatMul:product:0+cnn_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
cnn_2/output/SoftmaxSoftmaxcnn_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
m
IdentityIdentitycnn_2/output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp$^cnn_2/conv_1/BiasAdd/ReadVariableOp#^cnn_2/conv_1/Conv2D/ReadVariableOp$^cnn_2/conv_2/BiasAdd/ReadVariableOp#^cnn_2/conv_2/Conv2D/ReadVariableOp$^cnn_2/conv_3/BiasAdd/ReadVariableOp#^cnn_2/conv_3/Conv2D/ReadVariableOp%^cnn_2/dense_1/BiasAdd/ReadVariableOp$^cnn_2/dense_1/MatMul/ReadVariableOp$^cnn_2/output/BiasAdd/ReadVariableOp#^cnn_2/output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2J
#cnn_2/conv_1/BiasAdd/ReadVariableOp#cnn_2/conv_1/BiasAdd/ReadVariableOp2H
"cnn_2/conv_1/Conv2D/ReadVariableOp"cnn_2/conv_1/Conv2D/ReadVariableOp2J
#cnn_2/conv_2/BiasAdd/ReadVariableOp#cnn_2/conv_2/BiasAdd/ReadVariableOp2H
"cnn_2/conv_2/Conv2D/ReadVariableOp"cnn_2/conv_2/Conv2D/ReadVariableOp2J
#cnn_2/conv_3/BiasAdd/ReadVariableOp#cnn_2/conv_3/BiasAdd/ReadVariableOp2H
"cnn_2/conv_3/Conv2D/ReadVariableOp"cnn_2/conv_3/Conv2D/ReadVariableOp2L
$cnn_2/dense_1/BiasAdd/ReadVariableOp$cnn_2/dense_1/BiasAdd/ReadVariableOp2J
#cnn_2/dense_1/MatMul/ReadVariableOp#cnn_2/dense_1/MatMul/ReadVariableOp2J
#cnn_2/output/BiasAdd/ReadVariableOp#cnn_2/output/BiasAdd/ReadVariableOp2H
"cnn_2/output/MatMul/ReadVariableOp"cnn_2/output/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
D
(__inference_flatten_layer_call_fn_302944

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_302557a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
^
B__inference_pool_2_layer_call_and_return_conditional_losses_302939

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????		@:W S
/
_output_shapes
:?????????		@
 
_user_specified_nameinputs
?
?
'__inference_conv_3_layer_call_fn_302908

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_302539w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????		@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302740
input_1'
conv_1_302711: 
conv_1_302713: '
conv_2_302717: @
conv_2_302719:@'
conv_3_302722:@@
conv_3_302724:@!
dense_1_302729:	?d
dense_1_302731:d
output_302734:d

output_302736:

identity??conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_1_302711conv_1_302713*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_302499?
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_302509?
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_302717conv_2_302719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_302522?
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_302722conv_3_302724*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_302539?
pool_2/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_302549?
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_302557?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_302729dense_1_302731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302570?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_302734output_302736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_302587v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_output_layer_call_and_return_conditional_losses_302587

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
B__inference_output_layer_call_and_return_conditional_losses_302990

inputs0
matmul_readvariableop_resource:d
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
&__inference_cnn_2_layer_call_fn_302796

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	?d
	unknown_6:d
	unknown_7:d

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_2_layer_call_and_return_conditional_losses_302594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?n
?
"__inference__traced_restore_303191
file_prefix8
assignvariableop_conv_1_kernel: ,
assignvariableop_1_conv_1_bias: :
 assignvariableop_2_conv_2_kernel: @,
assignvariableop_3_conv_2_bias:@:
 assignvariableop_4_conv_3_kernel:@@,
assignvariableop_5_conv_3_bias:@4
!assignvariableop_6_dense_1_kernel:	?d-
assignvariableop_7_dense_1_bias:d2
 assignvariableop_8_output_kernel:d
,
assignvariableop_9_output_bias:
&
assignvariableop_10_sgd_iter:	 '
assignvariableop_11_sgd_decay: /
%assignvariableop_12_sgd_learning_rate: *
 assignvariableop_13_sgd_momentum: #
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: H
.assignvariableop_18_sgd_conv_1_kernel_momentum: :
,assignvariableop_19_sgd_conv_1_bias_momentum: H
.assignvariableop_20_sgd_conv_2_kernel_momentum: @:
,assignvariableop_21_sgd_conv_2_bias_momentum:@H
.assignvariableop_22_sgd_conv_3_kernel_momentum:@@:
,assignvariableop_23_sgd_conv_3_bias_momentum:@B
/assignvariableop_24_sgd_dense_1_kernel_momentum:	?d;
-assignvariableop_25_sgd_dense_1_bias_momentum:d@
.assignvariableop_26_sgd_output_kernel_momentum:d
:
,assignvariableop_27_sgd_output_bias_momentum:

identity_29??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(conv_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_1/bias/.ATTRIBUTES/VARIABLE_VALUEB(conv_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_2/bias/.ATTRIBUTES/VARIABLE_VALUEB(conv_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKconv_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBKconv_2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBKconv_3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBLdense_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJdense_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBHout/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBFout/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sgd_conv_1_kernel_momentumIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_sgd_conv_1_bias_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_sgd_conv_2_kernel_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp,assignvariableop_21_sgd_conv_2_bias_momentumIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp.assignvariableop_22_sgd_conv_3_kernel_momentumIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_sgd_conv_3_bias_momentumIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_sgd_dense_1_kernel_momentumIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_sgd_dense_1_bias_momentumIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_sgd_output_kernel_momentumIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_sgd_output_bias_momentumIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
$__inference_signature_wrapper_302771
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	?d
	unknown_6:d
	unknown_7:d

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_302457o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
B__inference_conv_2_layer_call_and_return_conditional_losses_302522

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv_3_layer_call_and_return_conditional_losses_302539

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????		@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????		@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
C
'__inference_pool_2_layer_call_fn_302924

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_302478?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_pool_1_layer_call_and_return_conditional_losses_302874

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_302959

inputs
unknown:	?d
	unknown_0:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302570o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_output_layer_call_fn_302979

inputs
unknown:d

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_302587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?<
?
__inference__traced_save_303097
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_sgd_conv_1_kernel_momentum_read_readvariableop7
3savev2_sgd_conv_1_bias_momentum_read_readvariableop9
5savev2_sgd_conv_2_kernel_momentum_read_readvariableop7
3savev2_sgd_conv_2_bias_momentum_read_readvariableop9
5savev2_sgd_conv_3_kernel_momentum_read_readvariableop7
3savev2_sgd_conv_3_bias_momentum_read_readvariableop:
6savev2_sgd_dense_1_kernel_momentum_read_readvariableop8
4savev2_sgd_dense_1_bias_momentum_read_readvariableop9
5savev2_sgd_output_kernel_momentum_read_readvariableop7
3savev2_sgd_output_bias_momentum_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B(conv_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_1/bias/.ATTRIBUTES/VARIABLE_VALUEB(conv_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_2/bias/.ATTRIBUTES/VARIABLE_VALUEB(conv_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB&conv_3/bias/.ATTRIBUTES/VARIABLE_VALUEB)dense_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'dense_1/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBKconv_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBKconv_2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBKconv_3/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIconv_3/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBLdense_1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJdense_1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBHout/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBFout/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_sgd_conv_1_kernel_momentum_read_readvariableop3savev2_sgd_conv_1_bias_momentum_read_readvariableop5savev2_sgd_conv_2_kernel_momentum_read_readvariableop3savev2_sgd_conv_2_bias_momentum_read_readvariableop5savev2_sgd_conv_3_kernel_momentum_read_readvariableop3savev2_sgd_conv_3_bias_momentum_read_readvariableop6savev2_sgd_dense_1_kernel_momentum_read_readvariableop4savev2_sgd_dense_1_bias_momentum_read_readvariableop5savev2_sgd_output_kernel_momentum_read_readvariableop3savev2_sgd_output_bias_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@@:@:	?d:d:d
:
: : : : : : : : : : : @:@:@@:@:	?d:d:d
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	?d: 

_output_shapes
:d:$	 

_output_shapes

:d
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	?d: 

_output_shapes
:d:$ 

_output_shapes

:d
: 

_output_shapes
:
:

_output_shapes
: 
?
^
B__inference_pool_1_layer_call_and_return_conditional_losses_302466

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv_2_layer_call_fn_302888

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_302522w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
C
'__inference_pool_1_layer_call_fn_302864

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_302466?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_302950

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_1_layer_call_and_return_conditional_losses_302570

inputs1
matmul_readvariableop_resource:	?d-
biasadd_readvariableop_resource:d
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????da
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
&__inference_cnn_2_layer_call_fn_302617
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	?d
	unknown_6:d
	unknown_7:d

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_2_layer_call_and_return_conditional_losses_302594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?#
?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302594

inputs'
conv_1_302500: 
conv_1_302502: '
conv_2_302523: @
conv_2_302525:@'
conv_3_302540:@@
conv_3_302542:@!
dense_1_302571:	?d
dense_1_302573:d
output_302588:d

output_302590:

identity??conv_1/StatefulPartitionedCall?conv_2/StatefulPartitionedCall?conv_3/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_302500conv_1_302502*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_302499?
pool_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_1_layer_call_and_return_conditional_losses_302509?
conv_2/StatefulPartitionedCallStatefulPartitionedCallpool_1/PartitionedCall:output:0conv_2_302523conv_2_302525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_302522?
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_302540conv_3_302542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_302539?
pool_2/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_pool_2_layer_call_and_return_conditional_losses_302549?
flatten/PartitionedCallPartitionedCallpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_302557?
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_302571dense_1_302573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_302570?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_302588output_302590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_302587v
IdentityIdentity'output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_pool_1_layer_call_and_return_conditional_losses_302509

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_302557

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?

conv_1

pool_1

conv_2

conv_3

pool_2
flatten
dense_1
out
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
9iter
	:decay
;learning_rate
<momentummomentumumomentumvmomentumwmomentumxmomentumy momentumz-momentum{.momentum|3momentum}4momentum~"
	optimizer
f
0
1
2
3
4
 5
-6
.7
38
49"
trackable_list_wrapper
f
0
1
2
3
4
 5
-6
.7
38
49"
trackable_list_wrapper
 "
trackable_list_wrapper
?
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics

	variables
trainable_variables
regularization_losses
__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv_1/kernel
: 2conv_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':% @2conv_2/kernel
:@2conv_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%@@2conv_3/kernel
:@2conv_3/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
)	variables
*trainable_variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?d2dense_1/kernel
:d2dense_1/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
/	variables
0trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:d
2output/kernel
:
2output/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
5	variables
6trainable_variables
7regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
.
j0
k1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ltotal
	mcount
n	variables
o	keras_api"
_tf_keras_metric
^
	ptotal
	qcount
r
_fn_kwargs
s	variables
t	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
l0
m1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
-
s	variables"
_generic_user_object
2:0 2SGD/conv_1/kernel/momentum
$:" 2SGD/conv_1/bias/momentum
2:0 @2SGD/conv_2/kernel/momentum
$:"@2SGD/conv_2/bias/momentum
2:0@@2SGD/conv_3/kernel/momentum
$:"@2SGD/conv_3/bias/momentum
,:*	?d2SGD/dense_1/kernel/momentum
%:#d2SGD/dense_1/bias/momentum
*:(d
2SGD/output/kernel/momentum
$:"
2SGD/output/bias/momentum
?2?
&__inference_cnn_2_layer_call_fn_302617
&__inference_cnn_2_layer_call_fn_302796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302839
A__inference_cnn_2_layer_call_and_return_conditional_losses_302740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
!__inference__wrapped_model_302457input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv_1_layer_call_fn_302848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv_1_layer_call_and_return_conditional_losses_302859?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_pool_1_layer_call_fn_302864
'__inference_pool_1_layer_call_fn_302869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_pool_1_layer_call_and_return_conditional_losses_302874
B__inference_pool_1_layer_call_and_return_conditional_losses_302879?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv_2_layer_call_fn_302888?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv_2_layer_call_and_return_conditional_losses_302899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv_3_layer_call_fn_302908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv_3_layer_call_and_return_conditional_losses_302919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_pool_2_layer_call_fn_302924
'__inference_pool_2_layer_call_fn_302929?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_pool_2_layer_call_and_return_conditional_losses_302934
B__inference_pool_2_layer_call_and_return_conditional_losses_302939?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_302944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_302950?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_302959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_302970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_output_layer_call_fn_302979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_output_layer_call_and_return_conditional_losses_302990?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_302771input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_302457{
 -.348?5
.?+
)?&
input_1?????????
? "3?0
.
output_1"?
output_1?????????
?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302740m
 -.348?5
.?+
)?&
input_1?????????
? "%?"
?
0?????????

? ?
A__inference_cnn_2_layer_call_and_return_conditional_losses_302839l
 -.347?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????

? ?
&__inference_cnn_2_layer_call_fn_302617`
 -.348?5
.?+
)?&
input_1?????????
? "??????????
?
&__inference_cnn_2_layer_call_fn_302796_
 -.347?4
-?*
(?%
inputs?????????
? "??????????
?
B__inference_conv_1_layer_call_and_return_conditional_losses_302859l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
'__inference_conv_1_layer_call_fn_302848_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
B__inference_conv_2_layer_call_and_return_conditional_losses_302899l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
'__inference_conv_2_layer_call_fn_302888_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
B__inference_conv_3_layer_call_and_return_conditional_losses_302919l 7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????		@
? ?
'__inference_conv_3_layer_call_fn_302908_ 7?4
-?*
(?%
inputs?????????@
? " ??????????		@?
C__inference_dense_1_layer_call_and_return_conditional_losses_302970]-.0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????d
? |
(__inference_dense_1_layer_call_fn_302959P-.0?-
&?#
!?
inputs??????????
? "??????????d?
C__inference_flatten_layer_call_and_return_conditional_losses_302950a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
(__inference_flatten_layer_call_fn_302944T7?4
-?*
(?%
inputs?????????@
? "????????????
B__inference_output_layer_call_and_return_conditional_losses_302990\34/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????

? z
'__inference_output_layer_call_fn_302979O34/?,
%?"
 ?
inputs?????????d
? "??????????
?
B__inference_pool_1_layer_call_and_return_conditional_losses_302874?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_pool_1_layer_call_and_return_conditional_losses_302879h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
'__inference_pool_1_layer_call_fn_302864?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_pool_1_layer_call_fn_302869[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
B__inference_pool_2_layer_call_and_return_conditional_losses_302934?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
B__inference_pool_2_layer_call_and_return_conditional_losses_302939h7?4
-?*
(?%
inputs?????????		@
? "-?*
#? 
0?????????@
? ?
'__inference_pool_2_layer_call_fn_302924?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
'__inference_pool_2_layer_call_fn_302929[7?4
-?*
(?%
inputs?????????		@
? " ??????????@?
$__inference_signature_wrapper_302771?
 -.34C?@
? 
9?6
4
input_1)?&
input_1?????????"3?0
.
output_1"?
output_1?????????
