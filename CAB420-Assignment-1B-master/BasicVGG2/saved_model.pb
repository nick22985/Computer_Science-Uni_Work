Ŷ 
??
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
executor_typestring ?
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_92/kernel
}
$conv2d_92/kernel/Read/ReadVariableOpReadVariableOpconv2d_92/kernel*&
_output_shapes
:*
dtype0
t
conv2d_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_92/bias
m
"conv2d_92/bias/Read/ReadVariableOpReadVariableOpconv2d_92/bias*
_output_shapes
:*
dtype0
?
conv2d_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_93/kernel
}
$conv2d_93/kernel/Read/ReadVariableOpReadVariableOpconv2d_93/kernel*&
_output_shapes
:*
dtype0
t
conv2d_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_93/bias
m
"conv2d_93/bias/Read/ReadVariableOpReadVariableOpconv2d_93/bias*
_output_shapes
:*
dtype0
?
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma
?
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta
?
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean
?
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance
?
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_94/kernel
}
$conv2d_94/kernel/Read/ReadVariableOpReadVariableOpconv2d_94/kernel*&
_output_shapes
:*
dtype0
t
conv2d_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_94/bias
m
"conv2d_94/bias/Read/ReadVariableOpReadVariableOpconv2d_94/bias*
_output_shapes
:*
dtype0
?
conv2d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_95/kernel
}
$conv2d_95/kernel/Read/ReadVariableOpReadVariableOpconv2d_95/kernel*&
_output_shapes
:*
dtype0
t
conv2d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_95/bias
m
"conv2d_95/bias/Read/ReadVariableOpReadVariableOpconv2d_95/bias*
_output_shapes
:*
dtype0
?
batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_20/gamma
?
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_20/beta
?
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_20/moving_mean
?
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_20/moving_variance
?
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_96/kernel
}
$conv2d_96/kernel/Read/ReadVariableOpReadVariableOpconv2d_96/kernel*&
_output_shapes
: *
dtype0
t
conv2d_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_96/bias
m
"conv2d_96/bias/Read/ReadVariableOpReadVariableOpconv2d_96/bias*
_output_shapes
: *
dtype0
?
conv2d_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_97/kernel
}
$conv2d_97/kernel/Read/ReadVariableOpReadVariableOpconv2d_97/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_97/bias
m
"conv2d_97/bias/Read/ReadVariableOpReadVariableOpconv2d_97/bias*
_output_shapes
: *
dtype0
?
batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_21/gamma
?
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
_output_shapes
: *
dtype0
?
batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_21/beta
?
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
_output_shapes
: *
dtype0
?
"batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_21/moving_mean
?
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
_output_shapes
: *
dtype0
?
&batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_21/moving_variance
?
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
_output_shapes
: *
dtype0
|
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_33/kernel
u
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel* 
_output_shapes
:
??*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:?*
dtype0
{
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_34/kernel
t
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes
:	?@*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:@*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:@
*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
Adam/conv2d_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_92/kernel/m
?
+Adam/conv2d_92/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_92/bias/m
{
)Adam/conv2d_92/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_93/kernel/m
?
+Adam/conv2d_93/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_93/bias/m
{
)Adam/conv2d_93/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_19/gamma/m
?
7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_19/beta/m
?
6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_94/kernel/m
?
+Adam/conv2d_94/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_94/bias/m
{
)Adam/conv2d_94/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_95/kernel/m
?
+Adam/conv2d_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_95/bias/m
{
)Adam/conv2d_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_20/gamma/m
?
7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_20/beta/m
?
6Adam/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_96/kernel/m
?
+Adam/conv2d_96/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_96/bias/m
{
)Adam/conv2d_96/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_97/kernel/m
?
+Adam/conv2d_97/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_97/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_97/bias/m
{
)Adam/conv2d_97/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_97/bias/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_21/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_21/gamma/m
?
7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/m*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_21/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_21/beta/m
?
6Adam/batch_normalization_21/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/m*
_output_shapes
: *
dtype0
?
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/m
?
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/m
z
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_34/kernel/m
?
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:@
*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_92/kernel/v
?
+Adam/conv2d_92/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_92/bias/v
{
)Adam/conv2d_92/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_93/kernel/v
?
+Adam/conv2d_93/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_93/bias/v
{
)Adam/conv2d_93/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_19/gamma/v
?
7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_19/beta/v
?
6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_94/kernel/v
?
+Adam/conv2d_94/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_94/bias/v
{
)Adam/conv2d_94/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_95/kernel/v
?
+Adam/conv2d_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_95/bias/v
{
)Adam/conv2d_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_20/gamma/v
?
7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_20/beta/v
?
6Adam/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_96/kernel/v
?
+Adam/conv2d_96/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_96/bias/v
{
)Adam/conv2d_96/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_96/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_97/kernel/v
?
+Adam/conv2d_97/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_97/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_97/bias/v
{
)Adam/conv2d_97/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_97/bias/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_21/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_21/gamma/v
?
7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_21/gamma/v*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_21/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_21/beta/v
?
6Adam/batch_normalization_21/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_21/beta/v*
_output_shapes
: *
dtype0
?
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_33/kernel/v
?
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_33/bias/v
z
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_34/kernel/v
?
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:@
*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-9
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
?
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
?
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
R
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
R
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
h

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
h

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
?
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
r	variables
strainable_variables
t	keras_api
R
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
R
yregularization_losses
z	variables
{trainable_variables
|	keras_api
S
}regularization_losses
~	variables
trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
n
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?$m?%m?+m?,m??m?@m?Em?Fm?Lm?Mm?`m?am?fm?gm?mm?nm?	?m?	?m?	?m?	?m?	?m?	?m?v?v?$v?%v?+v?,v??v?@v?Ev?Fv?Lv?Mv?`v?av?fv?gv?mv?nv?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
0
1
$2
%3
+4
,5
-6
.7
?8
@9
E10
F11
L12
M13
N14
O15
`16
a17
f18
g19
m20
n21
o22
p23
?24
?25
?26
?27
?28
?29
?
0
1
$2
%3
+4
,5
?6
@7
E8
F9
L10
M11
`12
a13
f14
g15
m16
n17
?18
?19
?20
?21
?22
?23
?
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
	variables
trainable_variables
?layer_metrics
 
\Z
VARIABLE_VALUEconv2d_92/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_92/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
 regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
!	variables
"trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_93/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_93/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
?
&regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
'	variables
(trainable_variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1
-2
.3

+0
,1
?
/regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
0	variables
1trainable_variables
?layer_metrics
 
 
 
?
3regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
4	variables
5trainable_variables
?layer_metrics
 
 
 
?
7regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
8	variables
9trainable_variables
?layer_metrics
 
 
 
?
;regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
<	variables
=trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_94/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_94/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
?
Aregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
B	variables
Ctrainable_variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_95/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_95/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
?
Gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
H	variables
Itrainable_variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1
N2
O3

L0
M1
?
Pregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
Q	variables
Rtrainable_variables
?layer_metrics
 
 
 
?
Tregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
U	variables
Vtrainable_variables
?layer_metrics
 
 
 
?
Xregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
Y	variables
Ztrainable_variables
?layer_metrics
 
 
 
?
\regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
]	variables
^trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_96/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_96/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1

`0
a1
?
bregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
c	variables
dtrainable_variables
?layer_metrics
\Z
VARIABLE_VALUEconv2d_97/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_97/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

f0
g1
?
hregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
i	variables
jtrainable_variables
?layer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1
o2
p3

m0
n1
?
qregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
r	variables
strainable_variables
?layer_metrics
 
 
 
?
uregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
v	variables
wtrainable_variables
?layer_metrics
 
 
 
?
yregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
z	variables
{trainable_variables
?layer_metrics
 
 
 
?
}regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
~	variables
trainable_variables
?layer_metrics
 
 
 
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEdense_34/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_34/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEdense_35/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_35/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
*
-0
.1
N2
O3
o4
p5
 
?
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
10
11
12
13
14
15
16
17
18
19
20
21
22

?0
?1
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

-0
.1
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

N0
O1
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

o0
p1
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_92/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_92/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_93/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_93/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_19/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_94/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_94/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_95/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_20/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_96/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_96/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_97/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_97/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_21/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_34/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_34/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_92/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_92/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_93/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_93/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_19/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_94/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_94/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_95/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_20/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_20/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_96/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_96/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_97/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_97/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_21/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_21/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_34/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_34/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_35/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_35/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_imgPlaceholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_imgconv2d_92/kernelconv2d_92/biasconv2d_93/kernelconv2d_93/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_96/kernelconv2d_96/biasconv2d_97/kernelconv2d_97/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_170897
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
? 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_92/kernel/Read/ReadVariableOp"conv2d_92/bias/Read/ReadVariableOp$conv2d_93/kernel/Read/ReadVariableOp"conv2d_93/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_94/kernel/Read/ReadVariableOp"conv2d_94/bias/Read/ReadVariableOp$conv2d_95/kernel/Read/ReadVariableOp"conv2d_95/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_96/kernel/Read/ReadVariableOp"conv2d_96/bias/Read/ReadVariableOp$conv2d_97/kernel/Read/ReadVariableOp"conv2d_97/bias/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_92/kernel/m/Read/ReadVariableOp)Adam/conv2d_92/bias/m/Read/ReadVariableOp+Adam/conv2d_93/kernel/m/Read/ReadVariableOp)Adam/conv2d_93/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp+Adam/conv2d_94/kernel/m/Read/ReadVariableOp)Adam/conv2d_94/bias/m/Read/ReadVariableOp+Adam/conv2d_95/kernel/m/Read/ReadVariableOp)Adam/conv2d_95/bias/m/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_20/beta/m/Read/ReadVariableOp+Adam/conv2d_96/kernel/m/Read/ReadVariableOp)Adam/conv2d_96/bias/m/Read/ReadVariableOp+Adam/conv2d_97/kernel/m/Read/ReadVariableOp)Adam/conv2d_97/bias/m/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_21/beta/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp+Adam/conv2d_92/kernel/v/Read/ReadVariableOp)Adam/conv2d_92/bias/v/Read/ReadVariableOp+Adam/conv2d_93/kernel/v/Read/ReadVariableOp)Adam/conv2d_93/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp+Adam/conv2d_94/kernel/v/Read/ReadVariableOp)Adam/conv2d_94/bias/v/Read/ReadVariableOp+Adam/conv2d_95/kernel/v/Read/ReadVariableOp)Adam/conv2d_95/bias/v/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_20/beta/v/Read/ReadVariableOp+Adam/conv2d_96/kernel/v/Read/ReadVariableOp)Adam/conv2d_96/bias/v/Read/ReadVariableOp+Adam/conv2d_97/kernel/v/Read/ReadVariableOp)Adam/conv2d_97/bias/v/Read/ReadVariableOp7Adam/batch_normalization_21/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_21/beta/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOpConst*d
Tin]
[2Y	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_172444
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_92/kernelconv2d_92/biasconv2d_93/kernelconv2d_93/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_96/kernelconv2d_96/biasconv2d_97/kernelconv2d_97/biasbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_variancedense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_92/kernel/mAdam/conv2d_92/bias/mAdam/conv2d_93/kernel/mAdam/conv2d_93/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/mAdam/conv2d_94/kernel/mAdam/conv2d_94/bias/mAdam/conv2d_95/kernel/mAdam/conv2d_95/bias/m#Adam/batch_normalization_20/gamma/m"Adam/batch_normalization_20/beta/mAdam/conv2d_96/kernel/mAdam/conv2d_96/bias/mAdam/conv2d_97/kernel/mAdam/conv2d_97/bias/m#Adam/batch_normalization_21/gamma/m"Adam/batch_normalization_21/beta/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/conv2d_92/kernel/vAdam/conv2d_92/bias/vAdam/conv2d_93/kernel/vAdam/conv2d_93/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/vAdam/conv2d_94/kernel/vAdam/conv2d_94/bias/vAdam/conv2d_95/kernel/vAdam/conv2d_95/bias/v#Adam/batch_normalization_20/gamma/v"Adam/batch_normalization_20/beta/vAdam/conv2d_96/kernel/vAdam/conv2d_96/bias/vAdam/conv2d_97/kernel/vAdam/conv2d_97/bias/v#Adam/batch_normalization_21/gamma/v"Adam/batch_normalization_21/beta/vAdam/dense_33/kernel/vAdam/dense_33/bias/vAdam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/v*c
Tin\
Z2X*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_172715??
?
e
I__inference_activation_20_layer_call_and_return_conditional_losses_172008

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_46_layer_call_fn_169609

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
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_1696032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Š
?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171197

inputs,
(conv2d_92_conv2d_readvariableop_resource-
)conv2d_92_biasadd_readvariableop_resource,
(conv2d_93_conv2d_readvariableop_resource-
)conv2d_93_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceC
?batch_normalization_20_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1?6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_20/ReadVariableOp?'batch_normalization_20/ReadVariableOp_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1? conv2d_92/BiasAdd/ReadVariableOp?conv2d_92/Conv2D/ReadVariableOp? conv2d_93/BiasAdd/ReadVariableOp?conv2d_93/Conv2D/ReadVariableOp? conv2d_94/BiasAdd/ReadVariableOp?conv2d_94/Conv2D/ReadVariableOp? conv2d_95/BiasAdd/ReadVariableOp?conv2d_95/Conv2D/ReadVariableOp? conv2d_96/BiasAdd/ReadVariableOp?conv2d_96/Conv2D/ReadVariableOp? conv2d_97/BiasAdd/ReadVariableOp?conv2d_97/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_92/Conv2D/ReadVariableOp?
conv2d_92/Conv2DConv2Dinputs'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_92/Conv2D?
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp?
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_92/Relu?
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_93/Conv2D/ReadVariableOp?
conv2d_93/Conv2DConv2Dconv2d_92/Relu:activations:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_93/Conv2D?
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp?
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_93/Relu?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_19/ReadVariableOp?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_93/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_19/FusedBatchNormV3?
activation_18/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
activation_18/Relu?
spatial_dropout2d_18/IdentityIdentity activation_18/Relu:activations:0*
T0*/
_output_shapes
:?????????  2
spatial_dropout2d_18/Identity?
max_pooling2d_45/MaxPoolMaxPool&spatial_dropout2d_18/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPool?
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_94/Conv2D/ReadVariableOp?
conv2d_94/Conv2DConv2D!max_pooling2d_45/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_94/Conv2D?
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp?
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_94/Relu?
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp?
conv2d_95/Conv2DConv2Dconv2d_94/Relu:activations:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_95/Conv2D?
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp?
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_95/Relu?
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_20/ReadVariableOp?
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_20/ReadVariableOp_1?
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_20/FusedBatchNormV3?
activation_19/ReluRelu+batch_normalization_20/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_19/Relu?
spatial_dropout2d_19/IdentityIdentity activation_19/Relu:activations:0*
T0*/
_output_shapes
:?????????2
spatial_dropout2d_19/Identity?
max_pooling2d_46/MaxPoolMaxPool&spatial_dropout2d_19/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPool?
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_96/Conv2D/ReadVariableOp?
conv2d_96/Conv2DConv2D!max_pooling2d_46/MaxPool:output:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_96/Conv2D?
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp?
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_96/Relu?
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_97/Conv2D/ReadVariableOp?
conv2d_97/Conv2DConv2Dconv2d_96/Relu:activations:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_97/Conv2D?
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp?
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_97/Relu?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_21/ReadVariableOp?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_21/ReadVariableOp_1?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_97/Relu:activations:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2)
'batch_normalization_21/FusedBatchNormV3?
activation_20/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation_20/Relu?
spatial_dropout2d_20/IdentityIdentity activation_20/Relu:activations:0*
T0*/
_output_shapes
:????????? 2
spatial_dropout2d_20/Identity?
max_pooling2d_47/MaxPoolMaxPool&spatial_dropout2d_20/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_15/Const?
flatten_15/ReshapeReshape!max_pooling2d_47/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_15/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_15/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAddt
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_33/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_34/Relu?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd|
dense_35/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_35/Softmax?

IdentityIdentitydense_35/Softmax:softmax:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp!^conv2d_97/BiasAdd/ReadVariableOp ^conv2d_97/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2D
 conv2d_97/BiasAdd/ReadVariableOp conv2d_97/BiasAdd/ReadVariableOp2B
conv2d_97/Conv2D/ReadVariableOpconv2d_97/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
ؙ
?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171077

inputs,
(conv2d_92_conv2d_readvariableop_resource-
)conv2d_92_biasadd_readvariableop_resource,
(conv2d_93_conv2d_readvariableop_resource-
)conv2d_93_biasadd_readvariableop_resource2
.batch_normalization_19_readvariableop_resource4
0batch_normalization_19_readvariableop_1_resourceC
?batch_normalization_19_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_94_conv2d_readvariableop_resource-
)conv2d_94_biasadd_readvariableop_resource,
(conv2d_95_conv2d_readvariableop_resource-
)conv2d_95_biasadd_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceC
?batch_normalization_20_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_96_conv2d_readvariableop_resource-
)conv2d_96_biasadd_readvariableop_resource,
(conv2d_97_conv2d_readvariableop_resource-
)conv2d_97_biasadd_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??%batch_normalization_19/AssignNewValue?'batch_normalization_19/AssignNewValue_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1?%batch_normalization_20/AssignNewValue?'batch_normalization_20/AssignNewValue_1?6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_20/ReadVariableOp?'batch_normalization_20/ReadVariableOp_1?%batch_normalization_21/AssignNewValue?'batch_normalization_21/AssignNewValue_1?6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_21/ReadVariableOp?'batch_normalization_21/ReadVariableOp_1? conv2d_92/BiasAdd/ReadVariableOp?conv2d_92/Conv2D/ReadVariableOp? conv2d_93/BiasAdd/ReadVariableOp?conv2d_93/Conv2D/ReadVariableOp? conv2d_94/BiasAdd/ReadVariableOp?conv2d_94/Conv2D/ReadVariableOp? conv2d_95/BiasAdd/ReadVariableOp?conv2d_95/Conv2D/ReadVariableOp? conv2d_96/BiasAdd/ReadVariableOp?conv2d_96/Conv2D/ReadVariableOp? conv2d_97/BiasAdd/ReadVariableOp?conv2d_97/Conv2D/ReadVariableOp?dense_33/BiasAdd/ReadVariableOp?dense_33/MatMul/ReadVariableOp?dense_34/BiasAdd/ReadVariableOp?dense_34/MatMul/ReadVariableOp?dense_35/BiasAdd/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_92/Conv2D/ReadVariableOp?
conv2d_92/Conv2DConv2Dinputs'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_92/Conv2D?
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp?
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_92/Relu?
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_93/Conv2D/ReadVariableOp?
conv2d_93/Conv2DConv2Dconv2d_92/Relu:activations:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_93/Conv2D?
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp?
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_93/Relu?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_19/ReadVariableOp?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_19/ReadVariableOp_1?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3conv2d_93/Relu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_19/FusedBatchNormV3?
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_19/AssignNewValue?
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_19/AssignNewValue_1?
activation_18/ReluRelu+batch_normalization_19/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
activation_18/Relu?
spatial_dropout2d_18/ShapeShape activation_18/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d_18/Shape?
(spatial_dropout2d_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(spatial_dropout2d_18/strided_slice/stack?
*spatial_dropout2d_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_18/strided_slice/stack_1?
*spatial_dropout2d_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_18/strided_slice/stack_2?
"spatial_dropout2d_18/strided_sliceStridedSlice#spatial_dropout2d_18/Shape:output:01spatial_dropout2d_18/strided_slice/stack:output:03spatial_dropout2d_18/strided_slice/stack_1:output:03spatial_dropout2d_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"spatial_dropout2d_18/strided_slice?
*spatial_dropout2d_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_18/strided_slice_1/stack?
,spatial_dropout2d_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_18/strided_slice_1/stack_1?
,spatial_dropout2d_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_18/strided_slice_1/stack_2?
$spatial_dropout2d_18/strided_slice_1StridedSlice#spatial_dropout2d_18/Shape:output:03spatial_dropout2d_18/strided_slice_1/stack:output:05spatial_dropout2d_18/strided_slice_1/stack_1:output:05spatial_dropout2d_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$spatial_dropout2d_18/strided_slice_1?
"spatial_dropout2d_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"spatial_dropout2d_18/dropout/Const?
 spatial_dropout2d_18/dropout/MulMul activation_18/Relu:activations:0+spatial_dropout2d_18/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  2"
 spatial_dropout2d_18/dropout/Mul?
3spatial_dropout2d_18/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_18/dropout/random_uniform/shape/1?
3spatial_dropout2d_18/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_18/dropout/random_uniform/shape/2?
1spatial_dropout2d_18/dropout/random_uniform/shapePack+spatial_dropout2d_18/strided_slice:output:0<spatial_dropout2d_18/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_18/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_18/strided_slice_1:output:0*
N*
T0*
_output_shapes
:23
1spatial_dropout2d_18/dropout/random_uniform/shape?
9spatial_dropout2d_18/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_18/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02;
9spatial_dropout2d_18/dropout/random_uniform/RandomUniform?
+spatial_dropout2d_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+spatial_dropout2d_18/dropout/GreaterEqual/y?
)spatial_dropout2d_18/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_18/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_18/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)spatial_dropout2d_18/dropout/GreaterEqual?
!spatial_dropout2d_18/dropout/CastCast-spatial_dropout2d_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2#
!spatial_dropout2d_18/dropout/Cast?
"spatial_dropout2d_18/dropout/Mul_1Mul$spatial_dropout2d_18/dropout/Mul:z:0%spatial_dropout2d_18/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  2$
"spatial_dropout2d_18/dropout/Mul_1?
max_pooling2d_45/MaxPoolMaxPool&spatial_dropout2d_18/dropout/Mul_1:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPool?
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_94/Conv2D/ReadVariableOp?
conv2d_94/Conv2DConv2D!max_pooling2d_45/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_94/Conv2D?
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp?
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_94/Relu?
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_95/Conv2D/ReadVariableOp?
conv2d_95/Conv2DConv2Dconv2d_94/Relu:activations:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_95/Conv2D?
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp?
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_95/Relu?
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_20/ReadVariableOp?
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_20/ReadVariableOp_1?
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_95/Relu:activations:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_20/FusedBatchNormV3?
%batch_normalization_20/AssignNewValueAssignVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource4batch_normalization_20/FusedBatchNormV3:batch_mean:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_20/AssignNewValue?
'batch_normalization_20/AssignNewValue_1AssignVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_20/FusedBatchNormV3:batch_variance:09^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_20/AssignNewValue_1?
activation_19/ReluRelu+batch_normalization_20/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
activation_19/Relu?
spatial_dropout2d_19/ShapeShape activation_19/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d_19/Shape?
(spatial_dropout2d_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(spatial_dropout2d_19/strided_slice/stack?
*spatial_dropout2d_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_19/strided_slice/stack_1?
*spatial_dropout2d_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_19/strided_slice/stack_2?
"spatial_dropout2d_19/strided_sliceStridedSlice#spatial_dropout2d_19/Shape:output:01spatial_dropout2d_19/strided_slice/stack:output:03spatial_dropout2d_19/strided_slice/stack_1:output:03spatial_dropout2d_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"spatial_dropout2d_19/strided_slice?
*spatial_dropout2d_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_19/strided_slice_1/stack?
,spatial_dropout2d_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_19/strided_slice_1/stack_1?
,spatial_dropout2d_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_19/strided_slice_1/stack_2?
$spatial_dropout2d_19/strided_slice_1StridedSlice#spatial_dropout2d_19/Shape:output:03spatial_dropout2d_19/strided_slice_1/stack:output:05spatial_dropout2d_19/strided_slice_1/stack_1:output:05spatial_dropout2d_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$spatial_dropout2d_19/strided_slice_1?
"spatial_dropout2d_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"spatial_dropout2d_19/dropout/Const?
 spatial_dropout2d_19/dropout/MulMul activation_19/Relu:activations:0+spatial_dropout2d_19/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2"
 spatial_dropout2d_19/dropout/Mul?
3spatial_dropout2d_19/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_19/dropout/random_uniform/shape/1?
3spatial_dropout2d_19/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_19/dropout/random_uniform/shape/2?
1spatial_dropout2d_19/dropout/random_uniform/shapePack+spatial_dropout2d_19/strided_slice:output:0<spatial_dropout2d_19/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_19/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_19/strided_slice_1:output:0*
N*
T0*
_output_shapes
:23
1spatial_dropout2d_19/dropout/random_uniform/shape?
9spatial_dropout2d_19/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_19/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02;
9spatial_dropout2d_19/dropout/random_uniform/RandomUniform?
+spatial_dropout2d_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+spatial_dropout2d_19/dropout/GreaterEqual/y?
)spatial_dropout2d_19/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_19/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_19/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)spatial_dropout2d_19/dropout/GreaterEqual?
!spatial_dropout2d_19/dropout/CastCast-spatial_dropout2d_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2#
!spatial_dropout2d_19/dropout/Cast?
"spatial_dropout2d_19/dropout/Mul_1Mul$spatial_dropout2d_19/dropout/Mul:z:0%spatial_dropout2d_19/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2$
"spatial_dropout2d_19/dropout/Mul_1?
max_pooling2d_46/MaxPoolMaxPool&spatial_dropout2d_19/dropout/Mul_1:z:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPool?
conv2d_96/Conv2D/ReadVariableOpReadVariableOp(conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_96/Conv2D/ReadVariableOp?
conv2d_96/Conv2DConv2D!max_pooling2d_46/MaxPool:output:0'conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_96/Conv2D?
 conv2d_96/BiasAdd/ReadVariableOpReadVariableOp)conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_96/BiasAdd/ReadVariableOp?
conv2d_96/BiasAddBiasAddconv2d_96/Conv2D:output:0(conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_96/BiasAdd~
conv2d_96/ReluReluconv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_96/Relu?
conv2d_97/Conv2D/ReadVariableOpReadVariableOp(conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_97/Conv2D/ReadVariableOp?
conv2d_97/Conv2DConv2Dconv2d_96/Relu:activations:0'conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_97/Conv2D?
 conv2d_97/BiasAdd/ReadVariableOpReadVariableOp)conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_97/BiasAdd/ReadVariableOp?
conv2d_97/BiasAddBiasAddconv2d_97/Conv2D:output:0(conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_97/BiasAdd~
conv2d_97/ReluReluconv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_97/Relu?
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_21/ReadVariableOp?
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_21/ReadVariableOp_1?
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_97/Relu:activations:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_21/FusedBatchNormV3?
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*R
_classH
FDloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_21/AssignNewValue?
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*T
_classJ
HFloc:@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_21/AssignNewValue_1?
activation_20/ReluRelu+batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
activation_20/Relu?
spatial_dropout2d_20/ShapeShape activation_20/Relu:activations:0*
T0*
_output_shapes
:2
spatial_dropout2d_20/Shape?
(spatial_dropout2d_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(spatial_dropout2d_20/strided_slice/stack?
*spatial_dropout2d_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_20/strided_slice/stack_1?
*spatial_dropout2d_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_20/strided_slice/stack_2?
"spatial_dropout2d_20/strided_sliceStridedSlice#spatial_dropout2d_20/Shape:output:01spatial_dropout2d_20/strided_slice/stack:output:03spatial_dropout2d_20/strided_slice/stack_1:output:03spatial_dropout2d_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"spatial_dropout2d_20/strided_slice?
*spatial_dropout2d_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*spatial_dropout2d_20/strided_slice_1/stack?
,spatial_dropout2d_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_20/strided_slice_1/stack_1?
,spatial_dropout2d_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,spatial_dropout2d_20/strided_slice_1/stack_2?
$spatial_dropout2d_20/strided_slice_1StridedSlice#spatial_dropout2d_20/Shape:output:03spatial_dropout2d_20/strided_slice_1/stack:output:05spatial_dropout2d_20/strided_slice_1/stack_1:output:05spatial_dropout2d_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$spatial_dropout2d_20/strided_slice_1?
"spatial_dropout2d_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"spatial_dropout2d_20/dropout/Const?
 spatial_dropout2d_20/dropout/MulMul activation_20/Relu:activations:0+spatial_dropout2d_20/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2"
 spatial_dropout2d_20/dropout/Mul?
3spatial_dropout2d_20/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_20/dropout/random_uniform/shape/1?
3spatial_dropout2d_20/dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :25
3spatial_dropout2d_20/dropout/random_uniform/shape/2?
1spatial_dropout2d_20/dropout/random_uniform/shapePack+spatial_dropout2d_20/strided_slice:output:0<spatial_dropout2d_20/dropout/random_uniform/shape/1:output:0<spatial_dropout2d_20/dropout/random_uniform/shape/2:output:0-spatial_dropout2d_20/strided_slice_1:output:0*
N*
T0*
_output_shapes
:23
1spatial_dropout2d_20/dropout/random_uniform/shape?
9spatial_dropout2d_20/dropout/random_uniform/RandomUniformRandomUniform:spatial_dropout2d_20/dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02;
9spatial_dropout2d_20/dropout/random_uniform/RandomUniform?
+spatial_dropout2d_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+spatial_dropout2d_20/dropout/GreaterEqual/y?
)spatial_dropout2d_20/dropout/GreaterEqualGreaterEqualBspatial_dropout2d_20/dropout/random_uniform/RandomUniform:output:04spatial_dropout2d_20/dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2+
)spatial_dropout2d_20/dropout/GreaterEqual?
!spatial_dropout2d_20/dropout/CastCast-spatial_dropout2d_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2#
!spatial_dropout2d_20/dropout/Cast?
"spatial_dropout2d_20/dropout/Mul_1Mul$spatial_dropout2d_20/dropout/Mul:z:0%spatial_dropout2d_20/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2$
"spatial_dropout2d_20/dropout/Mul_1?
max_pooling2d_47/MaxPoolMaxPool&spatial_dropout2d_20/dropout/Mul_1:z:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_15/Const?
flatten_15/ReshapeReshape!max_pooling2d_47/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_15/Reshape?
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_33/MatMul/ReadVariableOp?
dense_33/MatMulMatMulflatten_15/Reshape:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/MatMul?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_33/BiasAddt
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_33/Relu?
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_34/BiasAdds
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_34/Relu?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_35/BiasAdd|
dense_35/SoftmaxSoftmaxdense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_35/Softmax?
IdentityIdentitydense_35/Softmax:softmax:0&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1&^batch_normalization_20/AssignNewValue(^batch_normalization_20/AssignNewValue_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp!^conv2d_96/BiasAdd/ReadVariableOp ^conv2d_96/Conv2D/ReadVariableOp!^conv2d_97/BiasAdd/ReadVariableOp ^conv2d_97/Conv2D/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12N
%batch_normalization_20/AssignNewValue%batch_normalization_20/AssignNewValue2R
'batch_normalization_20/AssignNewValue_1'batch_normalization_20/AssignNewValue_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2D
 conv2d_96/BiasAdd/ReadVariableOp conv2d_96/BiasAdd/ReadVariableOp2B
conv2d_96/Conv2D/ReadVariableOpconv2d_96/Conv2D/ReadVariableOp2D
 conv2d_97/BiasAdd/ReadVariableOp conv2d_97/BiasAdd/ReadVariableOp2B
conv2d_97/Conv2D/ReadVariableOpconv2d_97/Conv2D/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_21_layer_call_fn_171926

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1702302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_169959

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
*__inference_simpleVGG_layer_call_fn_170671
img
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simpleVGG_layer_call_and_return_conditional_losses_1706082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
n
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_170144

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_19_layer_call_fn_171482

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1693032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_spatial_dropout2d_18_layer_call_fn_171543

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
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1694102
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv2d_97_layer_call_fn_171875

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1701952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_169768

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171787

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?f
?

E__inference_simpleVGG_layer_call_and_return_conditional_losses_170759

inputs
conv2d_92_170676
conv2d_92_170678
conv2d_93_170681
conv2d_93_170683!
batch_normalization_19_170686!
batch_normalization_19_170688!
batch_normalization_19_170690!
batch_normalization_19_170692
conv2d_94_170698
conv2d_94_170700
conv2d_95_170703
conv2d_95_170705!
batch_normalization_20_170708!
batch_normalization_20_170710!
batch_normalization_20_170712!
batch_normalization_20_170714
conv2d_96_170720
conv2d_96_170722
conv2d_97_170725
conv2d_97_170727!
batch_normalization_21_170730!
batch_normalization_21_170732!
batch_normalization_21_170734!
batch_normalization_21_170736
dense_33_170743
dense_33_170745
dense_34_170748
dense_34_170750
dense_35_170753
dense_35_170755
identity??.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_92/StatefulPartitionedCall?!conv2d_93/StatefulPartitionedCall?!conv2d_94/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall?!conv2d_97/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_92_170676conv2d_92_170678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1698082#
!conv2d_92/StatefulPartitionedCall?
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0conv2d_93_170681conv2d_93_170683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1698352#
!conv2d_93/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_19_170686batch_normalization_19_170688batch_normalization_19_170690batch_normalization_19_170692*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_16988820
.batch_normalization_19/StatefulPartitionedCall?
activation_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_1699292
activation_18/PartitionedCall?
$spatial_dropout2d_18/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699642&
$spatial_dropout2d_18/PartitionedCall?
 max_pooling2d_45/PartitionedCallPartitionedCall-spatial_dropout2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_1694192"
 max_pooling2d_45/PartitionedCall?
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_94_170698conv2d_94_170700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1699882#
!conv2d_94/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0conv2d_95_170703conv2d_95_170705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1700152#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_20_170708batch_normalization_20_170710batch_normalization_20_170712batch_normalization_20_170714*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_17006820
.batch_normalization_20/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_1701092
activation_19/PartitionedCall?
$spatial_dropout2d_19/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701442&
$spatial_dropout2d_19/PartitionedCall?
 max_pooling2d_46/PartitionedCallPartitionedCall-spatial_dropout2d_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_1696032"
 max_pooling2d_46/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_96_170720conv2d_96_170722*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1701682#
!conv2d_96/StatefulPartitionedCall?
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0conv2d_97_170725conv2d_97_170727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1701952#
!conv2d_97/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_21_170730batch_normalization_21_170732batch_normalization_21_170734batch_normalization_21_170736*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_17024820
.batch_normalization_21/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1702892
activation_20/PartitionedCall?
$spatial_dropout2d_20/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703242&
$spatial_dropout2d_20/PartitionedCall?
 max_pooling2d_47/PartitionedCallPartitionedCall-spatial_dropout2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_1697872"
 max_pooling2d_47/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1703432
flatten_15/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_33_170743dense_33_170745*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_1703622"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_170748dense_34_170750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_1703892"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_170753dense_35_170755*
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_1704162"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
D__inference_dense_33_layer_call_and_return_conditional_losses_170362

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172036

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_169964

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_170897
img
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1692412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
Q
5__inference_spatial_dropout2d_20_layer_call_fn_172051

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
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1697782
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_35_layer_call_and_return_conditional_losses_170416

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_19_layer_call_fn_171495

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1693342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171782

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv2d_96_layer_call_fn_171855

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1701682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171977

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_18_layer_call_fn_171538

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1694002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv2d_95_layer_call_fn_171621

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1700152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_activation_18_layer_call_and_return_conditional_losses_169929

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_169419

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_19_layer_call_fn_171792

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_activation_19_layer_call_fn_171759

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_1701092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_169518

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_170319

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_20_layer_call_fn_172084

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_18_layer_call_and_return_conditional_losses_171500

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
G
+__inference_flatten_15_layer_call_fn_172100

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
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1703432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_20_layer_call_fn_171736

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1700502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_conv2d_93_layer_call_fn_171367

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1698352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_169487

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_169603

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_169303

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172079

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_92_layer_call_and_return_conditional_losses_171338

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_19_layer_call_fn_171431

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1698882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_169241
img6
2simplevgg_conv2d_92_conv2d_readvariableop_resource7
3simplevgg_conv2d_92_biasadd_readvariableop_resource6
2simplevgg_conv2d_93_conv2d_readvariableop_resource7
3simplevgg_conv2d_93_biasadd_readvariableop_resource<
8simplevgg_batch_normalization_19_readvariableop_resource>
:simplevgg_batch_normalization_19_readvariableop_1_resourceM
Isimplevgg_batch_normalization_19_fusedbatchnormv3_readvariableop_resourceO
Ksimplevgg_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource6
2simplevgg_conv2d_94_conv2d_readvariableop_resource7
3simplevgg_conv2d_94_biasadd_readvariableop_resource6
2simplevgg_conv2d_95_conv2d_readvariableop_resource7
3simplevgg_conv2d_95_biasadd_readvariableop_resource<
8simplevgg_batch_normalization_20_readvariableop_resource>
:simplevgg_batch_normalization_20_readvariableop_1_resourceM
Isimplevgg_batch_normalization_20_fusedbatchnormv3_readvariableop_resourceO
Ksimplevgg_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource6
2simplevgg_conv2d_96_conv2d_readvariableop_resource7
3simplevgg_conv2d_96_biasadd_readvariableop_resource6
2simplevgg_conv2d_97_conv2d_readvariableop_resource7
3simplevgg_conv2d_97_biasadd_readvariableop_resource<
8simplevgg_batch_normalization_21_readvariableop_resource>
:simplevgg_batch_normalization_21_readvariableop_1_resourceM
Isimplevgg_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceO
Ksimplevgg_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource5
1simplevgg_dense_33_matmul_readvariableop_resource6
2simplevgg_dense_33_biasadd_readvariableop_resource5
1simplevgg_dense_34_matmul_readvariableop_resource6
2simplevgg_dense_34_biasadd_readvariableop_resource5
1simplevgg_dense_35_matmul_readvariableop_resource6
2simplevgg_dense_35_biasadd_readvariableop_resource
identity??@simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?BsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?/simpleVGG/batch_normalization_19/ReadVariableOp?1simpleVGG/batch_normalization_19/ReadVariableOp_1?@simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp?BsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?/simpleVGG/batch_normalization_20/ReadVariableOp?1simpleVGG/batch_normalization_20/ReadVariableOp_1?@simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?BsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?/simpleVGG/batch_normalization_21/ReadVariableOp?1simpleVGG/batch_normalization_21/ReadVariableOp_1?*simpleVGG/conv2d_92/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_92/Conv2D/ReadVariableOp?*simpleVGG/conv2d_93/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_93/Conv2D/ReadVariableOp?*simpleVGG/conv2d_94/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_94/Conv2D/ReadVariableOp?*simpleVGG/conv2d_95/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_95/Conv2D/ReadVariableOp?*simpleVGG/conv2d_96/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_96/Conv2D/ReadVariableOp?*simpleVGG/conv2d_97/BiasAdd/ReadVariableOp?)simpleVGG/conv2d_97/Conv2D/ReadVariableOp?)simpleVGG/dense_33/BiasAdd/ReadVariableOp?(simpleVGG/dense_33/MatMul/ReadVariableOp?)simpleVGG/dense_34/BiasAdd/ReadVariableOp?(simpleVGG/dense_34/MatMul/ReadVariableOp?)simpleVGG/dense_35/BiasAdd/ReadVariableOp?(simpleVGG/dense_35/MatMul/ReadVariableOp?
)simpleVGG/conv2d_92/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)simpleVGG/conv2d_92/Conv2D/ReadVariableOp?
simpleVGG/conv2d_92/Conv2DConv2Dimg1simpleVGG/conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
simpleVGG/conv2d_92/Conv2D?
*simpleVGG/conv2d_92/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simpleVGG/conv2d_92/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_92/BiasAddBiasAdd#simpleVGG/conv2d_92/Conv2D:output:02simpleVGG/conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
simpleVGG/conv2d_92/BiasAdd?
simpleVGG/conv2d_92/ReluRelu$simpleVGG/conv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
simpleVGG/conv2d_92/Relu?
)simpleVGG/conv2d_93/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)simpleVGG/conv2d_93/Conv2D/ReadVariableOp?
simpleVGG/conv2d_93/Conv2DConv2D&simpleVGG/conv2d_92/Relu:activations:01simpleVGG/conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
simpleVGG/conv2d_93/Conv2D?
*simpleVGG/conv2d_93/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simpleVGG/conv2d_93/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_93/BiasAddBiasAdd#simpleVGG/conv2d_93/Conv2D:output:02simpleVGG/conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
simpleVGG/conv2d_93/BiasAdd?
simpleVGG/conv2d_93/ReluRelu$simpleVGG/conv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
simpleVGG/conv2d_93/Relu?
/simpleVGG/batch_normalization_19/ReadVariableOpReadVariableOp8simplevgg_batch_normalization_19_readvariableop_resource*
_output_shapes
:*
dtype021
/simpleVGG/batch_normalization_19/ReadVariableOp?
1simpleVGG/batch_normalization_19/ReadVariableOp_1ReadVariableOp:simplevgg_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:*
dtype023
1simpleVGG/batch_normalization_19/ReadVariableOp_1?
@simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpIsimplevgg_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?
BsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsimplevgg_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
BsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?
1simpleVGG/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3&simpleVGG/conv2d_93/Relu:activations:07simpleVGG/batch_normalization_19/ReadVariableOp:value:09simpleVGG/batch_normalization_19/ReadVariableOp_1:value:0HsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0JsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 23
1simpleVGG/batch_normalization_19/FusedBatchNormV3?
simpleVGG/activation_18/ReluRelu5simpleVGG/batch_normalization_19/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  2
simpleVGG/activation_18/Relu?
'simpleVGG/spatial_dropout2d_18/IdentityIdentity*simpleVGG/activation_18/Relu:activations:0*
T0*/
_output_shapes
:?????????  2)
'simpleVGG/spatial_dropout2d_18/Identity?
"simpleVGG/max_pooling2d_45/MaxPoolMaxPool0simpleVGG/spatial_dropout2d_18/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2$
"simpleVGG/max_pooling2d_45/MaxPool?
)simpleVGG/conv2d_94/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)simpleVGG/conv2d_94/Conv2D/ReadVariableOp?
simpleVGG/conv2d_94/Conv2DConv2D+simpleVGG/max_pooling2d_45/MaxPool:output:01simpleVGG/conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
simpleVGG/conv2d_94/Conv2D?
*simpleVGG/conv2d_94/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simpleVGG/conv2d_94/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_94/BiasAddBiasAdd#simpleVGG/conv2d_94/Conv2D:output:02simpleVGG/conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
simpleVGG/conv2d_94/BiasAdd?
simpleVGG/conv2d_94/ReluRelu$simpleVGG/conv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
simpleVGG/conv2d_94/Relu?
)simpleVGG/conv2d_95/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)simpleVGG/conv2d_95/Conv2D/ReadVariableOp?
simpleVGG/conv2d_95/Conv2DConv2D&simpleVGG/conv2d_94/Relu:activations:01simpleVGG/conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
simpleVGG/conv2d_95/Conv2D?
*simpleVGG/conv2d_95/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simpleVGG/conv2d_95/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_95/BiasAddBiasAdd#simpleVGG/conv2d_95/Conv2D:output:02simpleVGG/conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
simpleVGG/conv2d_95/BiasAdd?
simpleVGG/conv2d_95/ReluRelu$simpleVGG/conv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
simpleVGG/conv2d_95/Relu?
/simpleVGG/batch_normalization_20/ReadVariableOpReadVariableOp8simplevgg_batch_normalization_20_readvariableop_resource*
_output_shapes
:*
dtype021
/simpleVGG/batch_normalization_20/ReadVariableOp?
1simpleVGG/batch_normalization_20/ReadVariableOp_1ReadVariableOp:simplevgg_batch_normalization_20_readvariableop_1_resource*
_output_shapes
:*
dtype023
1simpleVGG/batch_normalization_20/ReadVariableOp_1?
@simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpIsimplevgg_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp?
BsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsimplevgg_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
BsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1?
1simpleVGG/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3&simpleVGG/conv2d_95/Relu:activations:07simpleVGG/batch_normalization_20/ReadVariableOp:value:09simpleVGG/batch_normalization_20/ReadVariableOp_1:value:0HsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0JsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 23
1simpleVGG/batch_normalization_20/FusedBatchNormV3?
simpleVGG/activation_19/ReluRelu5simpleVGG/batch_normalization_20/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
simpleVGG/activation_19/Relu?
'simpleVGG/spatial_dropout2d_19/IdentityIdentity*simpleVGG/activation_19/Relu:activations:0*
T0*/
_output_shapes
:?????????2)
'simpleVGG/spatial_dropout2d_19/Identity?
"simpleVGG/max_pooling2d_46/MaxPoolMaxPool0simpleVGG/spatial_dropout2d_19/Identity:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2$
"simpleVGG/max_pooling2d_46/MaxPool?
)simpleVGG/conv2d_96/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_96_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)simpleVGG/conv2d_96/Conv2D/ReadVariableOp?
simpleVGG/conv2d_96/Conv2DConv2D+simpleVGG/max_pooling2d_46/MaxPool:output:01simpleVGG/conv2d_96/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
simpleVGG/conv2d_96/Conv2D?
*simpleVGG/conv2d_96/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_96_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*simpleVGG/conv2d_96/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_96/BiasAddBiasAdd#simpleVGG/conv2d_96/Conv2D:output:02simpleVGG/conv2d_96/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
simpleVGG/conv2d_96/BiasAdd?
simpleVGG/conv2d_96/ReluRelu$simpleVGG/conv2d_96/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
simpleVGG/conv2d_96/Relu?
)simpleVGG/conv2d_97/Conv2D/ReadVariableOpReadVariableOp2simplevgg_conv2d_97_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)simpleVGG/conv2d_97/Conv2D/ReadVariableOp?
simpleVGG/conv2d_97/Conv2DConv2D&simpleVGG/conv2d_96/Relu:activations:01simpleVGG/conv2d_97/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
simpleVGG/conv2d_97/Conv2D?
*simpleVGG/conv2d_97/BiasAdd/ReadVariableOpReadVariableOp3simplevgg_conv2d_97_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*simpleVGG/conv2d_97/BiasAdd/ReadVariableOp?
simpleVGG/conv2d_97/BiasAddBiasAdd#simpleVGG/conv2d_97/Conv2D:output:02simpleVGG/conv2d_97/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
simpleVGG/conv2d_97/BiasAdd?
simpleVGG/conv2d_97/ReluRelu$simpleVGG/conv2d_97/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
simpleVGG/conv2d_97/Relu?
/simpleVGG/batch_normalization_21/ReadVariableOpReadVariableOp8simplevgg_batch_normalization_21_readvariableop_resource*
_output_shapes
: *
dtype021
/simpleVGG/batch_normalization_21/ReadVariableOp?
1simpleVGG/batch_normalization_21/ReadVariableOp_1ReadVariableOp:simplevgg_batch_normalization_21_readvariableop_1_resource*
_output_shapes
: *
dtype023
1simpleVGG/batch_normalization_21/ReadVariableOp_1?
@simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpIsimplevgg_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02B
@simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp?
BsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsimplevgg_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02D
BsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1?
1simpleVGG/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3&simpleVGG/conv2d_97/Relu:activations:07simpleVGG/batch_normalization_21/ReadVariableOp:value:09simpleVGG/batch_normalization_21/ReadVariableOp_1:value:0HsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0JsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 23
1simpleVGG/batch_normalization_21/FusedBatchNormV3?
simpleVGG/activation_20/ReluRelu5simpleVGG/batch_normalization_21/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? 2
simpleVGG/activation_20/Relu?
'simpleVGG/spatial_dropout2d_20/IdentityIdentity*simpleVGG/activation_20/Relu:activations:0*
T0*/
_output_shapes
:????????? 2)
'simpleVGG/spatial_dropout2d_20/Identity?
"simpleVGG/max_pooling2d_47/MaxPoolMaxPool0simpleVGG/spatial_dropout2d_20/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2$
"simpleVGG/max_pooling2d_47/MaxPool?
simpleVGG/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
simpleVGG/flatten_15/Const?
simpleVGG/flatten_15/ReshapeReshape+simpleVGG/max_pooling2d_47/MaxPool:output:0#simpleVGG/flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
simpleVGG/flatten_15/Reshape?
(simpleVGG/dense_33/MatMul/ReadVariableOpReadVariableOp1simplevgg_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(simpleVGG/dense_33/MatMul/ReadVariableOp?
simpleVGG/dense_33/MatMulMatMul%simpleVGG/flatten_15/Reshape:output:00simpleVGG/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simpleVGG/dense_33/MatMul?
)simpleVGG/dense_33/BiasAdd/ReadVariableOpReadVariableOp2simplevgg_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)simpleVGG/dense_33/BiasAdd/ReadVariableOp?
simpleVGG/dense_33/BiasAddBiasAdd#simpleVGG/dense_33/MatMul:product:01simpleVGG/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
simpleVGG/dense_33/BiasAdd?
simpleVGG/dense_33/ReluRelu#simpleVGG/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
simpleVGG/dense_33/Relu?
(simpleVGG/dense_34/MatMul/ReadVariableOpReadVariableOp1simplevgg_dense_34_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02*
(simpleVGG/dense_34/MatMul/ReadVariableOp?
simpleVGG/dense_34/MatMulMatMul%simpleVGG/dense_33/Relu:activations:00simpleVGG/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simpleVGG/dense_34/MatMul?
)simpleVGG/dense_34/BiasAdd/ReadVariableOpReadVariableOp2simplevgg_dense_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)simpleVGG/dense_34/BiasAdd/ReadVariableOp?
simpleVGG/dense_34/BiasAddBiasAdd#simpleVGG/dense_34/MatMul:product:01simpleVGG/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
simpleVGG/dense_34/BiasAdd?
simpleVGG/dense_34/ReluRelu#simpleVGG/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
simpleVGG/dense_34/Relu?
(simpleVGG/dense_35/MatMul/ReadVariableOpReadVariableOp1simplevgg_dense_35_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02*
(simpleVGG/dense_35/MatMul/ReadVariableOp?
simpleVGG/dense_35/MatMulMatMul%simpleVGG/dense_34/Relu:activations:00simpleVGG/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simpleVGG/dense_35/MatMul?
)simpleVGG/dense_35/BiasAdd/ReadVariableOpReadVariableOp2simplevgg_dense_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)simpleVGG/dense_35/BiasAdd/ReadVariableOp?
simpleVGG/dense_35/BiasAddBiasAdd#simpleVGG/dense_35/MatMul:product:01simpleVGG/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
simpleVGG/dense_35/BiasAdd?
simpleVGG/dense_35/SoftmaxSoftmax#simpleVGG/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
simpleVGG/dense_35/Softmax?
IdentityIdentity$simpleVGG/dense_35/Softmax:softmax:0A^simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOpC^simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_10^simpleVGG/batch_normalization_19/ReadVariableOp2^simpleVGG/batch_normalization_19/ReadVariableOp_1A^simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOpC^simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_10^simpleVGG/batch_normalization_20/ReadVariableOp2^simpleVGG/batch_normalization_20/ReadVariableOp_1A^simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOpC^simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_10^simpleVGG/batch_normalization_21/ReadVariableOp2^simpleVGG/batch_normalization_21/ReadVariableOp_1+^simpleVGG/conv2d_92/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_92/Conv2D/ReadVariableOp+^simpleVGG/conv2d_93/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_93/Conv2D/ReadVariableOp+^simpleVGG/conv2d_94/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_94/Conv2D/ReadVariableOp+^simpleVGG/conv2d_95/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_95/Conv2D/ReadVariableOp+^simpleVGG/conv2d_96/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_96/Conv2D/ReadVariableOp+^simpleVGG/conv2d_97/BiasAdd/ReadVariableOp*^simpleVGG/conv2d_97/Conv2D/ReadVariableOp*^simpleVGG/dense_33/BiasAdd/ReadVariableOp)^simpleVGG/dense_33/MatMul/ReadVariableOp*^simpleVGG/dense_34/BiasAdd/ReadVariableOp)^simpleVGG/dense_34/MatMul/ReadVariableOp*^simpleVGG/dense_35/BiasAdd/ReadVariableOp)^simpleVGG/dense_35/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2?
@simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp@simpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
BsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1BsimpleVGG/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12b
/simpleVGG/batch_normalization_19/ReadVariableOp/simpleVGG/batch_normalization_19/ReadVariableOp2f
1simpleVGG/batch_normalization_19/ReadVariableOp_11simpleVGG/batch_normalization_19/ReadVariableOp_12?
@simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp@simpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2?
BsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1BsimpleVGG/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12b
/simpleVGG/batch_normalization_20/ReadVariableOp/simpleVGG/batch_normalization_20/ReadVariableOp2f
1simpleVGG/batch_normalization_20/ReadVariableOp_11simpleVGG/batch_normalization_20/ReadVariableOp_12?
@simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp@simpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2?
BsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1BsimpleVGG/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12b
/simpleVGG/batch_normalization_21/ReadVariableOp/simpleVGG/batch_normalization_21/ReadVariableOp2f
1simpleVGG/batch_normalization_21/ReadVariableOp_11simpleVGG/batch_normalization_21/ReadVariableOp_12X
*simpleVGG/conv2d_92/BiasAdd/ReadVariableOp*simpleVGG/conv2d_92/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_92/Conv2D/ReadVariableOp)simpleVGG/conv2d_92/Conv2D/ReadVariableOp2X
*simpleVGG/conv2d_93/BiasAdd/ReadVariableOp*simpleVGG/conv2d_93/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_93/Conv2D/ReadVariableOp)simpleVGG/conv2d_93/Conv2D/ReadVariableOp2X
*simpleVGG/conv2d_94/BiasAdd/ReadVariableOp*simpleVGG/conv2d_94/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_94/Conv2D/ReadVariableOp)simpleVGG/conv2d_94/Conv2D/ReadVariableOp2X
*simpleVGG/conv2d_95/BiasAdd/ReadVariableOp*simpleVGG/conv2d_95/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_95/Conv2D/ReadVariableOp)simpleVGG/conv2d_95/Conv2D/ReadVariableOp2X
*simpleVGG/conv2d_96/BiasAdd/ReadVariableOp*simpleVGG/conv2d_96/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_96/Conv2D/ReadVariableOp)simpleVGG/conv2d_96/Conv2D/ReadVariableOp2X
*simpleVGG/conv2d_97/BiasAdd/ReadVariableOp*simpleVGG/conv2d_97/BiasAdd/ReadVariableOp2V
)simpleVGG/conv2d_97/Conv2D/ReadVariableOp)simpleVGG/conv2d_97/Conv2D/ReadVariableOp2V
)simpleVGG/dense_33/BiasAdd/ReadVariableOp)simpleVGG/dense_33/BiasAdd/ReadVariableOp2T
(simpleVGG/dense_33/MatMul/ReadVariableOp(simpleVGG/dense_33/MatMul/ReadVariableOp2V
)simpleVGG/dense_34/BiasAdd/ReadVariableOp)simpleVGG/dense_34/BiasAdd/ReadVariableOp2T
(simpleVGG/dense_34/MatMul/ReadVariableOp(simpleVGG/dense_34/MatMul/ReadVariableOp2V
)simpleVGG/dense_35/BiasAdd/ReadVariableOp)simpleVGG/dense_35/BiasAdd/ReadVariableOp2T
(simpleVGG/dense_35/MatMul/ReadVariableOp(simpleVGG/dense_35/MatMul/ReadVariableOp:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
o
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171820

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_170343

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_169778

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
̫
?&
__inference__traced_save_172444
file_prefix/
+savev2_conv2d_92_kernel_read_readvariableop-
)savev2_conv2d_92_bias_read_readvariableop/
+savev2_conv2d_93_kernel_read_readvariableop-
)savev2_conv2d_93_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_94_kernel_read_readvariableop-
)savev2_conv2d_94_bias_read_readvariableop/
+savev2_conv2d_95_kernel_read_readvariableop-
)savev2_conv2d_95_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_96_kernel_read_readvariableop-
)savev2_conv2d_96_bias_read_readvariableop/
+savev2_conv2d_97_kernel_read_readvariableop-
)savev2_conv2d_97_bias_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_92_kernel_m_read_readvariableop4
0savev2_adam_conv2d_92_bias_m_read_readvariableop6
2savev2_adam_conv2d_93_kernel_m_read_readvariableop4
0savev2_adam_conv2d_93_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableop6
2savev2_adam_conv2d_94_kernel_m_read_readvariableop4
0savev2_adam_conv2d_94_bias_m_read_readvariableop6
2savev2_adam_conv2d_95_kernel_m_read_readvariableop4
0savev2_adam_conv2d_95_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_m_read_readvariableop6
2savev2_adam_conv2d_96_kernel_m_read_readvariableop4
0savev2_adam_conv2d_96_bias_m_read_readvariableop6
2savev2_adam_conv2d_97_kernel_m_read_readvariableop4
0savev2_adam_conv2d_97_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop6
2savev2_adam_conv2d_92_kernel_v_read_readvariableop4
0savev2_adam_conv2d_92_bias_v_read_readvariableop6
2savev2_adam_conv2d_93_kernel_v_read_readvariableop4
0savev2_adam_conv2d_93_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableop6
2savev2_adam_conv2d_94_kernel_v_read_readvariableop4
0savev2_adam_conv2d_94_bias_v_read_readvariableop6
2savev2_adam_conv2d_95_kernel_v_read_readvariableop4
0savev2_adam_conv2d_95_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_v_read_readvariableop6
2savev2_adam_conv2d_96_kernel_v_read_readvariableop4
0savev2_adam_conv2d_96_bias_v_read_readvariableop6
2savev2_adam_conv2d_97_kernel_v_read_readvariableop4
0savev2_adam_conv2d_97_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_21_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_21_beta_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?0
value?0B?0XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_92_kernel_read_readvariableop)savev2_conv2d_92_bias_read_readvariableop+savev2_conv2d_93_kernel_read_readvariableop)savev2_conv2d_93_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_94_kernel_read_readvariableop)savev2_conv2d_94_bias_read_readvariableop+savev2_conv2d_95_kernel_read_readvariableop)savev2_conv2d_95_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_96_kernel_read_readvariableop)savev2_conv2d_96_bias_read_readvariableop+savev2_conv2d_97_kernel_read_readvariableop)savev2_conv2d_97_bias_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_92_kernel_m_read_readvariableop0savev2_adam_conv2d_92_bias_m_read_readvariableop2savev2_adam_conv2d_93_kernel_m_read_readvariableop0savev2_adam_conv2d_93_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableop2savev2_adam_conv2d_94_kernel_m_read_readvariableop0savev2_adam_conv2d_94_bias_m_read_readvariableop2savev2_adam_conv2d_95_kernel_m_read_readvariableop0savev2_adam_conv2d_95_bias_m_read_readvariableop>savev2_adam_batch_normalization_20_gamma_m_read_readvariableop=savev2_adam_batch_normalization_20_beta_m_read_readvariableop2savev2_adam_conv2d_96_kernel_m_read_readvariableop0savev2_adam_conv2d_96_bias_m_read_readvariableop2savev2_adam_conv2d_97_kernel_m_read_readvariableop0savev2_adam_conv2d_97_bias_m_read_readvariableop>savev2_adam_batch_normalization_21_gamma_m_read_readvariableop=savev2_adam_batch_normalization_21_beta_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop2savev2_adam_conv2d_92_kernel_v_read_readvariableop0savev2_adam_conv2d_92_bias_v_read_readvariableop2savev2_adam_conv2d_93_kernel_v_read_readvariableop0savev2_adam_conv2d_93_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableop2savev2_adam_conv2d_94_kernel_v_read_readvariableop0savev2_adam_conv2d_94_bias_v_read_readvariableop2savev2_adam_conv2d_95_kernel_v_read_readvariableop0savev2_adam_conv2d_95_bias_v_read_readvariableop>savev2_adam_batch_normalization_20_gamma_v_read_readvariableop=savev2_adam_batch_normalization_20_beta_v_read_readvariableop2savev2_adam_conv2d_96_kernel_v_read_readvariableop0savev2_adam_conv2d_96_bias_v_read_readvariableop2savev2_adam_conv2d_97_kernel_v_read_readvariableop0savev2_adam_conv2d_97_bias_v_read_readvariableop>savev2_adam_batch_normalization_21_gamma_v_read_readvariableop=savev2_adam_batch_normalization_21_beta_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::::::: : :  : : : : : :
??:?:	?@:@:@
:
: : : : : : : : : ::::::::::::: : :  : : : :
??:?:	?@:@:@
:
::::::::::::: : :  : : : :
??:?:	?@:@:@
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
:: 2

_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: :,6(
&
_output_shapes
:  : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :&:"
 
_output_shapes
:
??:!;

_output_shapes	
:?:%<!

_output_shapes
:	?@: =

_output_shapes
:@:$> 

_output_shapes

:@
: ?

_output_shapes
:
:,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
:  : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :&R"
 
_output_shapes
:
??:!S

_output_shapes	
:?:%T!

_output_shapes
:	?@: U

_output_shapes
:@:$V 

_output_shapes

:@
: W

_output_shapes
:
:X

_output_shapes
: 
?l
?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170433
img
conv2d_92_169819
conv2d_92_169821
conv2d_93_169846
conv2d_93_169848!
batch_normalization_19_169915!
batch_normalization_19_169917!
batch_normalization_19_169919!
batch_normalization_19_169921
conv2d_94_169999
conv2d_94_170001
conv2d_95_170026
conv2d_95_170028!
batch_normalization_20_170095!
batch_normalization_20_170097!
batch_normalization_20_170099!
batch_normalization_20_170101
conv2d_96_170179
conv2d_96_170181
conv2d_97_170206
conv2d_97_170208!
batch_normalization_21_170275!
batch_normalization_21_170277!
batch_normalization_21_170279!
batch_normalization_21_170281
dense_33_170373
dense_33_170375
dense_34_170400
dense_34_170402
dense_35_170427
dense_35_170429
identity??.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_92/StatefulPartitionedCall?!conv2d_93/StatefulPartitionedCall?!conv2d_94/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall?!conv2d_97/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?,spatial_dropout2d_18/StatefulPartitionedCall?,spatial_dropout2d_19/StatefulPartitionedCall?,spatial_dropout2d_20/StatefulPartitionedCall?
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCallimgconv2d_92_169819conv2d_92_169821*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1698082#
!conv2d_92/StatefulPartitionedCall?
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0conv2d_93_169846conv2d_93_169848*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1698352#
!conv2d_93/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_19_169915batch_normalization_19_169917batch_normalization_19_169919batch_normalization_19_169921*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_16987020
.batch_normalization_19/StatefulPartitionedCall?
activation_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_1699292
activation_18/PartitionedCall?
,spatial_dropout2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699592.
,spatial_dropout2d_18/StatefulPartitionedCall?
 max_pooling2d_45/PartitionedCallPartitionedCall5spatial_dropout2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_1694192"
 max_pooling2d_45/PartitionedCall?
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_94_169999conv2d_94_170001*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1699882#
!conv2d_94/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0conv2d_95_170026conv2d_95_170028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1700152#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_20_170095batch_normalization_20_170097batch_normalization_20_170099batch_normalization_20_170101*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_17005020
.batch_normalization_20/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_1701092
activation_19/PartitionedCall?
,spatial_dropout2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0-^spatial_dropout2d_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701392.
,spatial_dropout2d_19/StatefulPartitionedCall?
 max_pooling2d_46/PartitionedCallPartitionedCall5spatial_dropout2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_1696032"
 max_pooling2d_46/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_96_170179conv2d_96_170181*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1701682#
!conv2d_96/StatefulPartitionedCall?
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0conv2d_97_170206conv2d_97_170208*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1701952#
!conv2d_97/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_21_170275batch_normalization_21_170277batch_normalization_21_170279batch_normalization_21_170281*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_17023020
.batch_normalization_21/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1702892
activation_20/PartitionedCall?
,spatial_dropout2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0-^spatial_dropout2d_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703192.
,spatial_dropout2d_20/StatefulPartitionedCall?
 max_pooling2d_47/PartitionedCallPartitionedCall5spatial_dropout2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_1697872"
 max_pooling2d_47/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1703432
flatten_15/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_33_170373dense_33_170375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_1703622"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_170400dense_34_170402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_1703892"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_170427dense_35_170429*
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_1704162"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall-^spatial_dropout2d_18/StatefulPartitionedCall-^spatial_dropout2d_19/StatefulPartitionedCall-^spatial_dropout2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2\
,spatial_dropout2d_18/StatefulPartitionedCall,spatial_dropout2d_18/StatefulPartitionedCall2\
,spatial_dropout2d_19/StatefulPartitionedCall,spatial_dropout2d_19/StatefulPartitionedCall2\
,spatial_dropout2d_20/StatefulPartitionedCall,spatial_dropout2d_20/StatefulPartitionedCall:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
M
1__inference_max_pooling2d_47_layer_call_fn_169793

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
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_1697872
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172074

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_18_layer_call_fn_171576

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
e
I__inference_activation_20_layer_call_and_return_conditional_losses_170289

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171566

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_19_layer_call_fn_171830

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1695842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_activation_19_layer_call_and_return_conditional_losses_171754

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_simpleVGG_layer_call_fn_171262

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simpleVGG_layer_call_and_return_conditional_losses_1706082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_93_layer_call_and_return_conditional_losses_169835

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
Q
5__inference_spatial_dropout2d_19_layer_call_fn_171797

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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701442
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_169888

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171913

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_20_layer_call_fn_171749

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1700682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_20_layer_call_fn_171685

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1695182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_20_layer_call_fn_171672

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_1694872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_169594

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_21_layer_call_fn_171939

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1702482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_34_layer_call_and_return_conditional_losses_172131

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_21_layer_call_fn_172003

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1697022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_34_layer_call_fn_172140

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_1703892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_170248

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171469

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_spatial_dropout2d_19_layer_call_fn_171835

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
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1695942
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171723

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_169870

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171571

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171895

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_169787

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171641

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_169702

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_96_layer_call_and_return_conditional_losses_171846

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_35_layer_call_and_return_conditional_losses_172151

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_169584

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_activation_18_layer_call_fn_171505

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
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_1699292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
~
)__inference_dense_35_layer_call_fn_172160

inputs
unknown
	unknown_0
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_1704162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_172095

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_170139

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_170230

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
E__inference_conv2d_94_layer_call_and_return_conditional_losses_171592

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_92_layer_call_and_return_conditional_losses_169808

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
E__inference_conv2d_95_layer_call_and_return_conditional_losses_170015

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?0
"__inference__traced_restore_172715
file_prefix%
!assignvariableop_conv2d_92_kernel%
!assignvariableop_1_conv2d_92_bias'
#assignvariableop_2_conv2d_93_kernel%
!assignvariableop_3_conv2d_93_bias3
/assignvariableop_4_batch_normalization_19_gamma2
.assignvariableop_5_batch_normalization_19_beta9
5assignvariableop_6_batch_normalization_19_moving_mean=
9assignvariableop_7_batch_normalization_19_moving_variance'
#assignvariableop_8_conv2d_94_kernel%
!assignvariableop_9_conv2d_94_bias(
$assignvariableop_10_conv2d_95_kernel&
"assignvariableop_11_conv2d_95_bias4
0assignvariableop_12_batch_normalization_20_gamma3
/assignvariableop_13_batch_normalization_20_beta:
6assignvariableop_14_batch_normalization_20_moving_mean>
:assignvariableop_15_batch_normalization_20_moving_variance(
$assignvariableop_16_conv2d_96_kernel&
"assignvariableop_17_conv2d_96_bias(
$assignvariableop_18_conv2d_97_kernel&
"assignvariableop_19_conv2d_97_bias4
0assignvariableop_20_batch_normalization_21_gamma3
/assignvariableop_21_batch_normalization_21_beta:
6assignvariableop_22_batch_normalization_21_moving_mean>
:assignvariableop_23_batch_normalization_21_moving_variance'
#assignvariableop_24_dense_33_kernel%
!assignvariableop_25_dense_33_bias'
#assignvariableop_26_dense_34_kernel%
!assignvariableop_27_dense_34_bias'
#assignvariableop_28_dense_35_kernel%
!assignvariableop_29_dense_35_bias!
assignvariableop_30_adam_iter#
assignvariableop_31_adam_beta_1#
assignvariableop_32_adam_beta_2"
assignvariableop_33_adam_decay*
&assignvariableop_34_adam_learning_rate
assignvariableop_35_total
assignvariableop_36_count
assignvariableop_37_total_1
assignvariableop_38_count_1/
+assignvariableop_39_adam_conv2d_92_kernel_m-
)assignvariableop_40_adam_conv2d_92_bias_m/
+assignvariableop_41_adam_conv2d_93_kernel_m-
)assignvariableop_42_adam_conv2d_93_bias_m;
7assignvariableop_43_adam_batch_normalization_19_gamma_m:
6assignvariableop_44_adam_batch_normalization_19_beta_m/
+assignvariableop_45_adam_conv2d_94_kernel_m-
)assignvariableop_46_adam_conv2d_94_bias_m/
+assignvariableop_47_adam_conv2d_95_kernel_m-
)assignvariableop_48_adam_conv2d_95_bias_m;
7assignvariableop_49_adam_batch_normalization_20_gamma_m:
6assignvariableop_50_adam_batch_normalization_20_beta_m/
+assignvariableop_51_adam_conv2d_96_kernel_m-
)assignvariableop_52_adam_conv2d_96_bias_m/
+assignvariableop_53_adam_conv2d_97_kernel_m-
)assignvariableop_54_adam_conv2d_97_bias_m;
7assignvariableop_55_adam_batch_normalization_21_gamma_m:
6assignvariableop_56_adam_batch_normalization_21_beta_m.
*assignvariableop_57_adam_dense_33_kernel_m,
(assignvariableop_58_adam_dense_33_bias_m.
*assignvariableop_59_adam_dense_34_kernel_m,
(assignvariableop_60_adam_dense_34_bias_m.
*assignvariableop_61_adam_dense_35_kernel_m,
(assignvariableop_62_adam_dense_35_bias_m/
+assignvariableop_63_adam_conv2d_92_kernel_v-
)assignvariableop_64_adam_conv2d_92_bias_v/
+assignvariableop_65_adam_conv2d_93_kernel_v-
)assignvariableop_66_adam_conv2d_93_bias_v;
7assignvariableop_67_adam_batch_normalization_19_gamma_v:
6assignvariableop_68_adam_batch_normalization_19_beta_v/
+assignvariableop_69_adam_conv2d_94_kernel_v-
)assignvariableop_70_adam_conv2d_94_bias_v/
+assignvariableop_71_adam_conv2d_95_kernel_v-
)assignvariableop_72_adam_conv2d_95_bias_v;
7assignvariableop_73_adam_batch_normalization_20_gamma_v:
6assignvariableop_74_adam_batch_normalization_20_beta_v/
+assignvariableop_75_adam_conv2d_96_kernel_v-
)assignvariableop_76_adam_conv2d_96_bias_v/
+assignvariableop_77_adam_conv2d_97_kernel_v-
)assignvariableop_78_adam_conv2d_97_bias_v;
7assignvariableop_79_adam_batch_normalization_21_gamma_v:
6assignvariableop_80_adam_batch_normalization_21_beta_v.
*assignvariableop_81_adam_dense_33_kernel_v,
(assignvariableop_82_adam_dense_33_bias_v.
*assignvariableop_83_adam_dense_34_kernel_v,
(assignvariableop_84_adam_dense_34_bias_v.
*assignvariableop_85_adam_dense_35_kernel_v,
(assignvariableop_86_adam_dense_35_bias_v
identity_88??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_9?1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?0
value?0B?0XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*?
value?B?XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_92_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_92_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_93_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_93_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_19_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_19_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_19_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_19_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_94_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_94_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_95_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_95_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_20_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_20_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_20_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_20_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_96_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_96_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_97_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_97_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_21_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_21_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_21_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_21_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_33_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_33_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_34_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_34_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_35_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_35_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_92_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_92_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_93_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_93_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_19_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_19_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_94_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_94_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_95_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_95_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_20_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_20_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_96_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_96_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_97_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_97_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_21_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_21_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_33_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_33_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_dense_34_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_dense_34_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_dense_35_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_dense_35_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_92_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_92_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_93_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_93_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_19_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_19_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_94_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_94_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_95_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_95_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_20_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_20_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_96_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_96_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_97_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_97_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_21_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_21_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_33_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_33_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_34_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_34_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_35_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_35_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_869
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_87?
Identity_88IdentityIdentity_87:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_88"#
identity_88Identity_88:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171659

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_19_layer_call_fn_171418

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_1698702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
Q
5__inference_spatial_dropout2d_18_layer_call_fn_171581

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
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172041

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_dense_33_layer_call_fn_172120

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_1703622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
o
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171528

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_93_layer_call_and_return_conditional_losses_171358

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

*__inference_conv2d_94_layer_call_fn_171601

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1699882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_97_layer_call_and_return_conditional_losses_171866

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_simpleVGG_layer_call_fn_170822
img
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simpleVGG_layer_call_and_return_conditional_losses_1707592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
o
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_169400

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const?
dropout/MulMulinputsdropout/Const:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul?
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1?
dropout/random_uniform/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/2?
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0'dropout/random_uniform/shape/2:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape?
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*8
_output_shapes&
$:"??????????????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*8
_output_shapes&
$:"??????????????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
dropout/Mul_1?
IdentityIdentitydropout/Mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171705

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_34_layer_call_and_return_conditional_losses_170389

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171959

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_170068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_96_layer_call_and_return_conditional_losses_170168

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_169410

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?f
?

E__inference_simpleVGG_layer_call_and_return_conditional_losses_170519
img
conv2d_92_170436
conv2d_92_170438
conv2d_93_170441
conv2d_93_170443!
batch_normalization_19_170446!
batch_normalization_19_170448!
batch_normalization_19_170450!
batch_normalization_19_170452
conv2d_94_170458
conv2d_94_170460
conv2d_95_170463
conv2d_95_170465!
batch_normalization_20_170468!
batch_normalization_20_170470!
batch_normalization_20_170472!
batch_normalization_20_170474
conv2d_96_170480
conv2d_96_170482
conv2d_97_170485
conv2d_97_170487!
batch_normalization_21_170490!
batch_normalization_21_170492!
batch_normalization_21_170494!
batch_normalization_21_170496
dense_33_170503
dense_33_170505
dense_34_170508
dense_34_170510
dense_35_170513
dense_35_170515
identity??.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_92/StatefulPartitionedCall?!conv2d_93/StatefulPartitionedCall?!conv2d_94/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall?!conv2d_97/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCallimgconv2d_92_170436conv2d_92_170438*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1698082#
!conv2d_92/StatefulPartitionedCall?
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0conv2d_93_170441conv2d_93_170443*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1698352#
!conv2d_93/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_19_170446batch_normalization_19_170448batch_normalization_19_170450batch_normalization_19_170452*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_16988820
.batch_normalization_19/StatefulPartitionedCall?
activation_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_1699292
activation_18/PartitionedCall?
$spatial_dropout2d_18/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699642&
$spatial_dropout2d_18/PartitionedCall?
 max_pooling2d_45/PartitionedCallPartitionedCall-spatial_dropout2d_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_1694192"
 max_pooling2d_45/PartitionedCall?
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_94_170458conv2d_94_170460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1699882#
!conv2d_94/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0conv2d_95_170463conv2d_95_170465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1700152#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_20_170468batch_normalization_20_170470batch_normalization_20_170472batch_normalization_20_170474*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_17006820
.batch_normalization_20/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_1701092
activation_19/PartitionedCall?
$spatial_dropout2d_19/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701442&
$spatial_dropout2d_19/PartitionedCall?
 max_pooling2d_46/PartitionedCallPartitionedCall-spatial_dropout2d_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_1696032"
 max_pooling2d_46/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_96_170480conv2d_96_170482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1701682#
!conv2d_96/StatefulPartitionedCall?
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0conv2d_97_170485conv2d_97_170487*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1701952#
!conv2d_97/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_21_170490batch_normalization_21_170492batch_normalization_21_170494batch_normalization_21_170496*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_17024820
.batch_normalization_21/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1702892
activation_20/PartitionedCall?
$spatial_dropout2d_20/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703242&
$spatial_dropout2d_20/PartitionedCall?
 max_pooling2d_47/PartitionedCallPartitionedCall-spatial_dropout2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_1697872"
 max_pooling2d_47/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1703432
flatten_15/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_33_170503dense_33_170505*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_1703622"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_170508dense_34_170510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_1703892"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_170513dense_35_170515*
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_1704162"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:T P
/
_output_shapes
:?????????  

_user_specified_nameimg
?
n
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171533

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv2d_92_layer_call_fn_171347

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1698082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_45_layer_call_fn_169425

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
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_1694192
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_170050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_95_layer_call_and_return_conditional_losses_171612

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_21_layer_call_fn_171990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1696712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
e
I__inference_activation_19_layer_call_and_return_conditional_losses_170109

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_170324

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171387

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
Q
5__inference_spatial_dropout2d_20_layer_call_fn_172089

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
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171405

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????  ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
n
5__inference_spatial_dropout2d_20_layer_call_fn_172046

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1697682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_169671

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_33_layer_call_and_return_conditional_losses_172111

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_169334

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_94_layer_call_and_return_conditional_losses_169988

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_activation_20_layer_call_fn_172013

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
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1702892
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?l
?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170608

inputs
conv2d_92_170525
conv2d_92_170527
conv2d_93_170530
conv2d_93_170532!
batch_normalization_19_170535!
batch_normalization_19_170537!
batch_normalization_19_170539!
batch_normalization_19_170541
conv2d_94_170547
conv2d_94_170549
conv2d_95_170552
conv2d_95_170554!
batch_normalization_20_170557!
batch_normalization_20_170559!
batch_normalization_20_170561!
batch_normalization_20_170563
conv2d_96_170569
conv2d_96_170571
conv2d_97_170574
conv2d_97_170576!
batch_normalization_21_170579!
batch_normalization_21_170581!
batch_normalization_21_170583!
batch_normalization_21_170585
dense_33_170592
dense_33_170594
dense_34_170597
dense_34_170599
dense_35_170602
dense_35_170604
identity??.batch_normalization_19/StatefulPartitionedCall?.batch_normalization_20/StatefulPartitionedCall?.batch_normalization_21/StatefulPartitionedCall?!conv2d_92/StatefulPartitionedCall?!conv2d_93/StatefulPartitionedCall?!conv2d_94/StatefulPartitionedCall?!conv2d_95/StatefulPartitionedCall?!conv2d_96/StatefulPartitionedCall?!conv2d_97/StatefulPartitionedCall? dense_33/StatefulPartitionedCall? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?,spatial_dropout2d_18/StatefulPartitionedCall?,spatial_dropout2d_19/StatefulPartitionedCall?,spatial_dropout2d_20/StatefulPartitionedCall?
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_92_170525conv2d_92_170527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1698082#
!conv2d_92/StatefulPartitionedCall?
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0conv2d_93_170530conv2d_93_170532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1698352#
!conv2d_93/StatefulPartitionedCall?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0batch_normalization_19_170535batch_normalization_19_170537batch_normalization_19_170539batch_normalization_19_170541*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_16987020
.batch_normalization_19/StatefulPartitionedCall?
activation_18/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_18_layer_call_and_return_conditional_losses_1699292
activation_18/PartitionedCall?
,spatial_dropout2d_18/StatefulPartitionedCallStatefulPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_1699592.
,spatial_dropout2d_18/StatefulPartitionedCall?
 max_pooling2d_45/PartitionedCallPartitionedCall5spatial_dropout2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_1694192"
 max_pooling2d_45/PartitionedCall?
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0conv2d_94_170547conv2d_94_170549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1699882#
!conv2d_94/StatefulPartitionedCall?
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0conv2d_95_170552conv2d_95_170554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1700152#
!conv2d_95/StatefulPartitionedCall?
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0batch_normalization_20_170557batch_normalization_20_170559batch_normalization_20_170561batch_normalization_20_170563*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_17005020
.batch_normalization_20/StatefulPartitionedCall?
activation_19/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_19_layer_call_and_return_conditional_losses_1701092
activation_19/PartitionedCall?
,spatial_dropout2d_19/StatefulPartitionedCallStatefulPartitionedCall&activation_19/PartitionedCall:output:0-^spatial_dropout2d_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_1701392.
,spatial_dropout2d_19/StatefulPartitionedCall?
 max_pooling2d_46/PartitionedCallPartitionedCall5spatial_dropout2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_1696032"
 max_pooling2d_46/PartitionedCall?
!conv2d_96/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0conv2d_96_170569conv2d_96_170571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_96_layer_call_and_return_conditional_losses_1701682#
!conv2d_96/StatefulPartitionedCall?
!conv2d_97/StatefulPartitionedCallStatefulPartitionedCall*conv2d_96/StatefulPartitionedCall:output:0conv2d_97_170574conv2d_97_170576*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_97_layer_call_and_return_conditional_losses_1701952#
!conv2d_97/StatefulPartitionedCall?
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_97/StatefulPartitionedCall:output:0batch_normalization_21_170579batch_normalization_21_170581batch_normalization_21_170583batch_normalization_21_170585*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_17023020
.batch_normalization_21/StatefulPartitionedCall?
activation_20/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_20_layer_call_and_return_conditional_losses_1702892
activation_20/PartitionedCall?
,spatial_dropout2d_20/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0-^spatial_dropout2d_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_1703192.
,spatial_dropout2d_20/StatefulPartitionedCall?
 max_pooling2d_47/PartitionedCallPartitionedCall5spatial_dropout2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_1697872"
 max_pooling2d_47/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1703432
flatten_15/PartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_33_170592dense_33_170594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_33_layer_call_and_return_conditional_losses_1703622"
 dense_33/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_170597dense_34_170599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_1703892"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_170602dense_35_170604*
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
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_1704162"
 dense_35/StatefulPartitionedCall?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall"^conv2d_96/StatefulPartitionedCall"^conv2d_97/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall-^spatial_dropout2d_18/StatefulPartitionedCall-^spatial_dropout2d_19/StatefulPartitionedCall-^spatial_dropout2d_20/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2F
!conv2d_96/StatefulPartitionedCall!conv2d_96/StatefulPartitionedCall2F
!conv2d_97/StatefulPartitionedCall!conv2d_97/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall2\
,spatial_dropout2d_18/StatefulPartitionedCall,spatial_dropout2d_18/StatefulPartitionedCall2\
,spatial_dropout2d_19/StatefulPartitionedCall,spatial_dropout2d_19/StatefulPartitionedCall2\
,spatial_dropout2d_20/StatefulPartitionedCall,spatial_dropout2d_20/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171451

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
n
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171825

inputs

identity_1}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity?

Identity_1IdentityIdentity:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity_1"!

identity_1Identity_1:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_97_layer_call_and_return_conditional_losses_170195

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_simpleVGG_layer_call_fn_171327

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_simpleVGG_layer_call_and_return_conditional_losses_1707592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????  ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
img4
serving_default_img:0?????????  <
dense_350
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
޹
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer-19
layer_with_weights-9
layer-20
layer_with_weights-10
layer-21
layer_with_weights-11
layer-22
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_network??{"class_name": "Functional", "name": "simpleVGG", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "simpleVGG", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "img"}, "name": "img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_92", "inbound_nodes": [[["img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_93", "inbound_nodes": [[["conv2d_92", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_93", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_18", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_45", "inbound_nodes": [[["spatial_dropout2d_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_94", "inbound_nodes": [[["max_pooling2d_45", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["conv2d_94", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_19", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_46", "inbound_nodes": [[["spatial_dropout2d_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["max_pooling2d_46", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_97", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_97", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_20", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_47", "inbound_nodes": [[["spatial_dropout2d_20", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["max_pooling2d_47", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["flatten_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}], "input_layers": [["img", 0, 0]], "output_layers": [["dense_35", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "simpleVGG", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "img"}, "name": "img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_92", "inbound_nodes": [[["img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_93", "inbound_nodes": [[["conv2d_92", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_19", "inbound_nodes": [[["conv2d_93", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_18", "inbound_nodes": [[["batch_normalization_19", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_18", "inbound_nodes": [[["activation_18", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_45", "inbound_nodes": [[["spatial_dropout2d_18", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_94", "inbound_nodes": [[["max_pooling2d_45", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_95", "inbound_nodes": [[["conv2d_94", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_95", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_19", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_19", "inbound_nodes": [[["activation_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_46", "inbound_nodes": [[["spatial_dropout2d_19", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_96", "inbound_nodes": [[["max_pooling2d_46", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_97", "inbound_nodes": [[["conv2d_96", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_97", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_20", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "SpatialDropout2D", "config": {"name": "spatial_dropout2d_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "spatial_dropout2d_20", "inbound_nodes": [[["activation_20", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_47", "inbound_nodes": [[["spatial_dropout2d_20", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["max_pooling2d_47", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["flatten_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_34", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_35", "inbound_nodes": [[["dense_34", 0, 0, {}]]]}], "input_layers": [["img", 0, 0]], "output_layers": [["dense_35", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "img", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "img"}}
?	

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_92", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_92", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
?	

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_93", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_93", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 8]}}
?	
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_19", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 8]}}
?
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;regularization_losses
<	variables
=trainable_variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_94", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_94", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 8]}}
?	

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_95", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_95", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
?	
Kaxis
	Lgamma
Mbeta
Nmoving_mean
Omoving_variance
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
?
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

`kernel
abias
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_96", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 16]}}
?	

fkernel
gbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_97", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
?	
laxis
	mgamma
nbeta
omoving_mean
pmoving_variance
qregularization_losses
r	variables
strainable_variables
t	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
?
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
yregularization_losses
z	variables
{trainable_variables
|	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SpatialDropout2D", "name": "spatial_dropout2d_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "spatial_dropout2d_20", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
}regularization_losses
~	variables
trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
?
?kernel
	?bias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?$m?%m?+m?,m??m?@m?Em?Fm?Lm?Mm?`m?am?fm?gm?mm?nm?	?m?	?m?	?m?	?m?	?m?	?m?v?v?$v?%v?+v?,v??v?@v?Ev?Fv?Lv?Mv?`v?av?fv?gv?mv?nv?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
0
1
$2
%3
+4
,5
-6
.7
?8
@9
E10
F11
L12
M13
N14
O15
`16
a17
f18
g19
m20
n21
o22
p23
?24
?25
?26
?27
?28
?29"
trackable_list_wrapper
?
0
1
$2
%3
+4
,5
?6
@7
E8
F9
L10
M11
`12
a13
f14
g15
m16
n17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?
regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
	variables
trainable_variables
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_92/kernel
:2conv2d_92/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
!	variables
"trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_93/kernel
:2conv2d_93/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
&regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
'	variables
(trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_19/gamma
):'2batch_normalization_19/beta
2:0 (2"batch_normalization_19/moving_mean
6:4 (2&batch_normalization_19/moving_variance
 "
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
/regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
0	variables
1trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
4	variables
5trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
8	variables
9trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
<	variables
=trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_94/kernel
:2conv2d_94/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
Aregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
B	variables
Ctrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_95/kernel
:2conv2d_95/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
Gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
H	variables
Itrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_20/gamma
):'2batch_normalization_20/beta
2:0 (2"batch_normalization_20/moving_mean
6:4 (2&batch_normalization_20/moving_variance
 "
trackable_list_wrapper
<
L0
M1
N2
O3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
Pregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
Q	variables
Rtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
U	variables
Vtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
Y	variables
Ztrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
]	variables
^trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_96/kernel
: 2conv2d_96/bias
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
?
bregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
c	variables
dtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_97/kernel
: 2conv2d_97/bias
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
?
hregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
i	variables
jtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_21/gamma
):' 2batch_normalization_21/beta
2:0  (2"batch_normalization_21/moving_mean
6:4  (2&batch_normalization_21/moving_variance
 "
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
qregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
r	variables
strainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
uregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
v	variables
wtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
yregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
z	variables
{trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
~	variables
trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_33/kernel
:?2dense_33/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_34/kernel
:@2dense_34/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@
2dense_35/kernel
:
2dense_35/bias
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layers
?metrics
?	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J
-0
.1
N2
O3
o4
p5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
0
?0
?1"
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
.
-0
.1"
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
.
N0
O1"
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
.
o0
p1"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/conv2d_92/kernel/m
!:2Adam/conv2d_92/bias/m
/:-2Adam/conv2d_93/kernel/m
!:2Adam/conv2d_93/bias/m
/:-2#Adam/batch_normalization_19/gamma/m
.:,2"Adam/batch_normalization_19/beta/m
/:-2Adam/conv2d_94/kernel/m
!:2Adam/conv2d_94/bias/m
/:-2Adam/conv2d_95/kernel/m
!:2Adam/conv2d_95/bias/m
/:-2#Adam/batch_normalization_20/gamma/m
.:,2"Adam/batch_normalization_20/beta/m
/:- 2Adam/conv2d_96/kernel/m
!: 2Adam/conv2d_96/bias/m
/:-  2Adam/conv2d_97/kernel/m
!: 2Adam/conv2d_97/bias/m
/:- 2#Adam/batch_normalization_21/gamma/m
.:, 2"Adam/batch_normalization_21/beta/m
(:&
??2Adam/dense_33/kernel/m
!:?2Adam/dense_33/bias/m
':%	?@2Adam/dense_34/kernel/m
 :@2Adam/dense_34/bias/m
&:$@
2Adam/dense_35/kernel/m
 :
2Adam/dense_35/bias/m
/:-2Adam/conv2d_92/kernel/v
!:2Adam/conv2d_92/bias/v
/:-2Adam/conv2d_93/kernel/v
!:2Adam/conv2d_93/bias/v
/:-2#Adam/batch_normalization_19/gamma/v
.:,2"Adam/batch_normalization_19/beta/v
/:-2Adam/conv2d_94/kernel/v
!:2Adam/conv2d_94/bias/v
/:-2Adam/conv2d_95/kernel/v
!:2Adam/conv2d_95/bias/v
/:-2#Adam/batch_normalization_20/gamma/v
.:,2"Adam/batch_normalization_20/beta/v
/:- 2Adam/conv2d_96/kernel/v
!: 2Adam/conv2d_96/bias/v
/:-  2Adam/conv2d_97/kernel/v
!: 2Adam/conv2d_97/bias/v
/:- 2#Adam/batch_normalization_21/gamma/v
.:, 2"Adam/batch_normalization_21/beta/v
(:&
??2Adam/dense_33/kernel/v
!:?2Adam/dense_33/bias/v
':%	?@2Adam/dense_34/kernel/v
 :@2Adam/dense_34/bias/v
&:$@
2Adam/dense_35/kernel/v
 :
2Adam/dense_35/bias/v
?2?
!__inference__wrapped_model_169241?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
img?????????  
?2?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171197
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170519
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170433
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171077?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_simpleVGG_layer_call_fn_170822
*__inference_simpleVGG_layer_call_fn_171327
*__inference_simpleVGG_layer_call_fn_171262
*__inference_simpleVGG_layer_call_fn_170671?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv2d_92_layer_call_and_return_conditional_losses_171338?
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
*__inference_conv2d_92_layer_call_fn_171347?
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
E__inference_conv2d_93_layer_call_and_return_conditional_losses_171358?
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
*__inference_conv2d_93_layer_call_fn_171367?
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
?2?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171405
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171387
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171469
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171451?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_19_layer_call_fn_171431
7__inference_batch_normalization_19_layer_call_fn_171495
7__inference_batch_normalization_19_layer_call_fn_171482
7__inference_batch_normalization_19_layer_call_fn_171418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_18_layer_call_and_return_conditional_losses_171500?
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
.__inference_activation_18_layer_call_fn_171505?
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
?2?
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171533
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171528
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171571
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171566?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_spatial_dropout2d_18_layer_call_fn_171576
5__inference_spatial_dropout2d_18_layer_call_fn_171581
5__inference_spatial_dropout2d_18_layer_call_fn_171538
5__inference_spatial_dropout2d_18_layer_call_fn_171543?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_169419?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_45_layer_call_fn_169425?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2d_94_layer_call_and_return_conditional_losses_171592?
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
*__inference_conv2d_94_layer_call_fn_171601?
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
E__inference_conv2d_95_layer_call_and_return_conditional_losses_171612?
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
*__inference_conv2d_95_layer_call_fn_171621?
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
?2?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171659
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171705
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171641
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171723?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_20_layer_call_fn_171736
7__inference_batch_normalization_20_layer_call_fn_171749
7__inference_batch_normalization_20_layer_call_fn_171685
7__inference_batch_normalization_20_layer_call_fn_171672?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_19_layer_call_and_return_conditional_losses_171754?
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
.__inference_activation_19_layer_call_fn_171759?
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
?2?
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171787
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171820
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171782
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171825?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_spatial_dropout2d_19_layer_call_fn_171835
5__inference_spatial_dropout2d_19_layer_call_fn_171797
5__inference_spatial_dropout2d_19_layer_call_fn_171830
5__inference_spatial_dropout2d_19_layer_call_fn_171792?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_169603?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_46_layer_call_fn_169609?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2d_96_layer_call_and_return_conditional_losses_171846?
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
*__inference_conv2d_96_layer_call_fn_171855?
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
E__inference_conv2d_97_layer_call_and_return_conditional_losses_171866?
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
*__inference_conv2d_97_layer_call_fn_171875?
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
?2?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171895
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171913
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171959
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171977?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_batch_normalization_21_layer_call_fn_171939
7__inference_batch_normalization_21_layer_call_fn_171990
7__inference_batch_normalization_21_layer_call_fn_171926
7__inference_batch_normalization_21_layer_call_fn_172003?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_activation_20_layer_call_and_return_conditional_losses_172008?
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
.__inference_activation_20_layer_call_fn_172013?
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
?2?
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172036
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172079
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172041
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172074?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_spatial_dropout2d_20_layer_call_fn_172089
5__inference_spatial_dropout2d_20_layer_call_fn_172046
5__inference_spatial_dropout2d_20_layer_call_fn_172084
5__inference_spatial_dropout2d_20_layer_call_fn_172051?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_169787?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_max_pooling2d_47_layer_call_fn_169793?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_flatten_15_layer_call_and_return_conditional_losses_172095?
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
+__inference_flatten_15_layer_call_fn_172100?
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
D__inference_dense_33_layer_call_and_return_conditional_losses_172111?
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
)__inference_dense_33_layer_call_fn_172120?
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
D__inference_dense_34_layer_call_and_return_conditional_losses_172131?
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
)__inference_dense_34_layer_call_fn_172140?
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
D__inference_dense_35_layer_call_and_return_conditional_losses_172151?
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
)__inference_dense_35_layer_call_fn_172160?
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
$__inference_signature_wrapper_170897img"?
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
!__inference__wrapped_model_169241?$$%+,-.?@EFLMNO`afgmnop??????4?1
*?'
%?"
img?????????  
? "3?0
.
dense_35"?
dense_35?????????
?
I__inference_activation_18_layer_call_and_return_conditional_losses_171500h7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
.__inference_activation_18_layer_call_fn_171505[7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
I__inference_activation_19_layer_call_and_return_conditional_losses_171754h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
.__inference_activation_19_layer_call_fn_171759[7?4
-?*
(?%
inputs?????????
? " ???????????
I__inference_activation_20_layer_call_and_return_conditional_losses_172008h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
.__inference_activation_20_layer_call_fn_172013[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171387r+,-.;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171405r+,-.;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171451?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_19_layer_call_and_return_conditional_losses_171469?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
7__inference_batch_normalization_19_layer_call_fn_171418e+,-.;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
7__inference_batch_normalization_19_layer_call_fn_171431e+,-.;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
7__inference_batch_normalization_19_layer_call_fn_171482?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_19_layer_call_fn_171495?+,-.M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171641?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171659?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171705rLMNO;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
R__inference_batch_normalization_20_layer_call_and_return_conditional_losses_171723rLMNO;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
7__inference_batch_normalization_20_layer_call_fn_171672?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_batch_normalization_20_layer_call_fn_171685?LMNOM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_batch_normalization_20_layer_call_fn_171736eLMNO;?8
1?.
(?%
inputs?????????
p
? " ???????????
7__inference_batch_normalization_20_layer_call_fn_171749eLMNO;?8
1?.
(?%
inputs?????????
p 
? " ???????????
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171895rmnop;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171913rmnop;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171959?mnopM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_batch_normalization_21_layer_call_and_return_conditional_losses_171977?mnopM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
7__inference_batch_normalization_21_layer_call_fn_171926emnop;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
7__inference_batch_normalization_21_layer_call_fn_171939emnop;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
7__inference_batch_normalization_21_layer_call_fn_171990?mnopM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_batch_normalization_21_layer_call_fn_172003?mnopM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
E__inference_conv2d_92_layer_call_and_return_conditional_losses_171338l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_92_layer_call_fn_171347_7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_93_layer_call_and_return_conditional_losses_171358l$%7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
*__inference_conv2d_93_layer_call_fn_171367_$%7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
E__inference_conv2d_94_layer_call_and_return_conditional_losses_171592l?@7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_94_layer_call_fn_171601_?@7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_conv2d_95_layer_call_and_return_conditional_losses_171612lEF7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_95_layer_call_fn_171621_EF7?4
-?*
(?%
inputs?????????
? " ???????????
E__inference_conv2d_96_layer_call_and_return_conditional_losses_171846l`a7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_96_layer_call_fn_171855_`a7?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_97_layer_call_and_return_conditional_losses_171866lfg7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_97_layer_call_fn_171875_fg7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
D__inference_dense_33_layer_call_and_return_conditional_losses_172111`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
)__inference_dense_33_layer_call_fn_172120S??0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_34_layer_call_and_return_conditional_losses_172131_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? 
)__inference_dense_34_layer_call_fn_172140R??0?-
&?#
!?
inputs??????????
? "??????????@?
D__inference_dense_35_layer_call_and_return_conditional_losses_172151^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????

? ~
)__inference_dense_35_layer_call_fn_172160Q??/?,
%?"
 ?
inputs?????????@
? "??????????
?
F__inference_flatten_15_layer_call_and_return_conditional_losses_172095a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
+__inference_flatten_15_layer_call_fn_172100T7?4
-?*
(?%
inputs????????? 
? "????????????
L__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_169419?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_45_layer_call_fn_169425?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_169603?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_46_layer_call_fn_169609?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_169787?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_47_layer_call_fn_169793?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_170897?$$%+,-.?@EFLMNO`afgmnop??????;?8
? 
1?.
,
img%?"
img?????????  "3?0
.
dense_35"?
dense_35?????????
?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170433?$$%+,-.?@EFLMNO`afgmnop??????<?9
2?/
%?"
img?????????  
p

 
? "%?"
?
0?????????

? ?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_170519?$$%+,-.?@EFLMNO`afgmnop??????<?9
2?/
%?"
img?????????  
p 

 
? "%?"
?
0?????????

? ?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171077?$$%+,-.?@EFLMNO`afgmnop????????<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
E__inference_simpleVGG_layer_call_and_return_conditional_losses_171197?$$%+,-.?@EFLMNO`afgmnop????????<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
*__inference_simpleVGG_layer_call_fn_170671~$$%+,-.?@EFLMNO`afgmnop??????<?9
2?/
%?"
img?????????  
p

 
? "??????????
?
*__inference_simpleVGG_layer_call_fn_170822~$$%+,-.?@EFLMNO`afgmnop??????<?9
2?/
%?"
img?????????  
p 

 
? "??????????
?
*__inference_simpleVGG_layer_call_fn_171262?$$%+,-.?@EFLMNO`afgmnop????????<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
*__inference_simpleVGG_layer_call_fn_171327?$$%+,-.?@EFLMNO`afgmnop????????<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171528?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171533?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171566l;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
P__inference_spatial_dropout2d_18_layer_call_and_return_conditional_losses_171571l;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
5__inference_spatial_dropout2d_18_layer_call_fn_171538?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
5__inference_spatial_dropout2d_18_layer_call_fn_171543?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
5__inference_spatial_dropout2d_18_layer_call_fn_171576_;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
5__inference_spatial_dropout2d_18_layer_call_fn_171581_;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171782l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171787l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171820?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_spatial_dropout2d_19_layer_call_and_return_conditional_losses_171825?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
5__inference_spatial_dropout2d_19_layer_call_fn_171792_;?8
1?.
(?%
inputs?????????
p
? " ???????????
5__inference_spatial_dropout2d_19_layer_call_fn_171797_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
5__inference_spatial_dropout2d_19_layer_call_fn_171830?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
5__inference_spatial_dropout2d_19_layer_call_fn_171835?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172036?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172041?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? "H?E
>?;
04????????????????????????????????????
? ?
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172074l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
P__inference_spatial_dropout2d_20_layer_call_and_return_conditional_losses_172079l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
5__inference_spatial_dropout2d_20_layer_call_fn_172046?V?S
L?I
C?@
inputs4????????????????????????????????????
p
? ";?84?????????????????????????????????????
5__inference_spatial_dropout2d_20_layer_call_fn_172051?V?S
L?I
C?@
inputs4????????????????????????????????????
p 
? ";?84?????????????????????????????????????
5__inference_spatial_dropout2d_20_layer_call_fn_172084_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
5__inference_spatial_dropout2d_20_layer_call_fn_172089_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? 