import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_regularizer

class SigmoidRange(nn.Module):
	'''
	Shape:
		- Input: :math:`(*)`, where :math:`*` means any number of dimensions.
		- Output: :math:`(*)`, same shape as the input.


	Examples::

		>>> m = nn.Sigmoid()
		>>> input = torch.randn(2)
		>>> output = m(input)
	'''

	def __init__(self, low, high):
		super(SigmoidRange, self).__init__()
		self.low  = low
		self.high = high

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		return torch.sigmoid(input) * (self.high - self.low) + self.low 


'''
class PointNetHead(nn.Module):
	def __init__(self, num_classes, encoder_channels):
		super().__init__()
		self.fc1 = nn.Linear(encoder_channels, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 128)

		# Classification head
		self.classify = nn.Sequential(nn.Linear(128, num_classes), nn.LogSoftmax(dim=1))
		
		# Regression heads
		self.regress_x = nn.Linear(128, 1) 
		self.regress_y = nn.Linear(128, 1)
		self.regress_z = nn.Linear(128, 1)
		
		# Angle classification head
		self.angle_class = nn.Sequential(nn.Linear(128, 4), nn.LogSoftmax(dim=1))

	def forward(self, x):
		x = F.relu(self.fc1(x)) 
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return {
			"classify": self.classify(x),
			"regress_x": self.regress_x(x),
			"regress_y": self.regress_y(x),
			"regress_z": self.regress_z(x),
			"angle_class": self.angle_class(x)
		}
'''


class ClassificationHead(nn.Module):
	def __init__(self, num_classes=2, num_rot_classes=6, num_type_classes=2, debug=False):
		super().__init__()
		self.debug = debug
		self.num_classes = num_classes
		self.num_rot_classes = num_rot_classes
		self.num_type_classes = num_type_classes

		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, num_classes)      # 2 classes for binary label	(e.g. astroid, geom_petal)
		self.fc4 = nn.Linear(64, num_rot_classes)  # 6 classes for rot		(e.g. [-1, π/5, π/4, π/3, π/2, π] so we can encode them just as [0, 5, 4, 3, 2, 1])
		self.fc5 = nn.Linear(64, num_type_classes) # 2 classes for type		(axis, plane)

		if self.debug:
			print(f'ClassificationHead.__init__() - num_classes: {num_classes} - num_rot_classes: {num_rot_classes} - num_type_classes: {num_type_classes}')

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.3)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, p=0.3)
		cls = F.softmax(self.fc3(x), dim=1)
		rot = F.softmax(self.fc4(x), dim=1)
		typ = F.softmax(self.fc5(x), dim=1)

		if self.debug:
			print(f'ClassificationHead.forward() - cls: {cls} - rot: {rot} - typ: {typ}')
		return cls, rot, typ

'''
We have to predict something like this:

gt_arr: [0.582, 0.418, 0.744]								# popx, popy, popz	(just three float for each point cloud)
gt_mat: [
		[-0.125, -0.544, -0.474, 0.296, -0.829, -0.829, -0.544, -0.125],	# nx * 8		(except that it's not *8, it seems *14 as of today)
		[0.220, 0.799, 0.721, -0.409, -0.558, -0.558, 0.799, 0.220],		# ny * 8
		[-0.967, 0.253, -0.505, -0.862, -0.019, -0.019, 0.253, -0.967]		# nz * 8
	]
gt_cat: [
		[0, 0, 0, 0, 1, 0, 1, 1],						# type * 8
		[nan, nan, nan, nan, 1.570, nan, 3.141, 3.141]				# rot  * 8
	]
gt_cls: 0										# cls (astroid, geom_petal, etc.)

'''

class RegressionHead(nn.Module):
	def __init__(self, pop_floats=3, normal_floats=3, normal_max_rows=14, debug=False):
		super().__init__()
		self.debug = debug
		self.pop_floats = pop_floats
		self.normal_floats = normal_floats
		self.normal_max_rows = normal_max_rows

		self.fc1 = nn.Linear(256, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, self.pop_floats)					# 3 float values for popx, popy, popz
		self.fc4 = nn.Linear(64, self.normal_floats * self.normal_max_rows)		# 42 float values for nx, ny, nz (*14 rows)

		if self.debug:
			print(f'RegressionHead.__init__() - pop_floats: {pop_floats} - normal_floats: {normal_floats} - normal_max_rows: {normal_max_rows}')

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=0.3)
		x = F.relu(self.fc2(x))
		x = F.dropout(x, p=0.3)

		pop  = self.fc3(x)
		norm = self.fc4(x)

		norm_flat = norm.view(-1, self.normal_floats * self.normal_max_rows)

		if self.debug:
			print(f'RegressionHead.forward() - pop: {pop} - norm: {norm} - norm_flat: {norm_flat}')

		return pop, norm_flat



class MLPRegressionHead(nn.Module):
	def __init__(self, in_dim, hidden_dim='unused-previously-256', out_dim=2048, pop_floats=-1, normal_floats=-1, normal_max_rows=-1, debug=False):
		super().__init__()

		self.debug = debug
		self.pop_floats = pop_floats
		self.normal_floats = normal_floats
		self.normal_max_rows = normal_max_rows

		if pop_floats != -1:
			self.name = 'pop'
		else:
			self.name = 'norm'

		'''
		self.layer1 = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.Dropout(p=0.1),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim//2),
			nn.BatchNorm1d(hidden_dim//2),
			nn.Dropout(p=0.1),
			nn.ReLU(inplace=True)
		)
		self.layer3 = nn.Sequential(
			nn.Linear(hidden_dim//2, hidden_dim//4),
			nn.BatchNorm1d(hidden_dim//4),
			nn.Dropout(p=0.1),
			nn.ReLU(inplace=True)
		)
		self.layer4 = nn.Sequential(
			nn.Linear(hidden_dim//4, hidden_dim//2),
			nn.BatchNorm1d(hidden_dim//2),
			nn.Dropout(p=0.1),
			nn.ReLU(inplace=True)
		)
		self.layer5 = nn.Sequential(
			nn.Linear(hidden_dim//2, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.Dropout(p=0.1),
			nn.ReLU(inplace=True)
		)
		'''


		'''
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25)
    (4): Linear(in_features=1024, out_features=512, bias=True)
    (5): ReLU(inplace)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5)
    (8): Linear(in_features=512, out_features=37, bias=True)			
		'''
		self.layer_fastai = nn.Sequential(
			nn.BatchNorm1d(in_dim),
			#nn.Dropout(p=0.25),
			nn.Linear(in_dim, in_dim//2),
			nn.ReLU(inplace=True)
		)

		if pop_floats != -1:
			out_dim = self.pop_floats
		else:
			out_dim = self.normal_floats * self.normal_max_rows
		'''
		self.layer_out = nn.Sequential(
			nn.Dropout(p=0.1),
			nn.Linear(hidden_dim, out_dim),
			nn.BatchNorm1d(out_dim)
		)
		'''

		self.layer_fastaiout = nn.Sequential(
			nn.BatchNorm1d(in_dim//2),
			#nn.Dropout(p=0.5),
			nn.Linear(in_dim//2, out_dim),
		)

		#self.num_layers = 3

		if self.debug:
			print(f'MLPRegressionHead.__init__() - pop_floats: {pop_floats} - normal_floats: {normal_floats} - normal_max_rows: {normal_max_rows}')

	def set_layers(self, num_layers):
		self.num_layers = num_layers

	def forward(self, x):
		'''
		if self.num_layers == 3:
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
		elif self.num_layers == 2:
			x = self.layer1(x)
			x = self.layer3(x)
		else:
			raise Exception
		'''

		'''
		x   = self.layer1(x)
		x   = self.layer2(x)
		x   = self.layer3(x)
		x   = self.layer4(x)
		x   = self.layer5(x)
		out = self.layer_out(x)
		'''
		x   = self.layer_fastai(x)
		out = self.layer_fastaiout(x)

		if self.debug:
			print(f'MLPRegressionHead.forward() - {self.name} - out.shape : {out.shape } - out : {out}')
		'''
		out = out.permute(1, 0).contiguous()
		if self.debug:
			print(f'MLPRegressionHead.forward() - {self.name} - out.shape : {out.shape } - out : {out}')
		'''

		return out 



class RegressionModel_(nn.Module):
	#def __init__(self, num_features_in, conv_features_out=256, pop_floats=3, normal_floats=3, normal_max_rows=14, debug=False):
	#def __init__(self, num_features_in, conv_features_out=256, pop_floats=-1, normal_floats=-1, normal_max_rows=-1, debug=False):
	def __init__(self, num_features_in, conv_features_out=256, pop_floats=-1, normal_floats=-1, normal_max_rows=-1, debug=False):
		super(RegressionModel, self).__init__()

		self.debug = debug
		self.pop_floats = pop_floats
		self.normal_floats = normal_floats
		self.normal_max_rows = normal_max_rows

		if pop_floats != -1:
			self.name = 'pop'
		else:
			self.name = 'norm'

		'''
 92         self.conv1 = torch.nn.Conv1d(channel, 64, 1)
 93         self.conv2 = torch.nn.Conv1d(64, 128, 1)
 94         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
 95         self.bn1 = nn.BatchNorm1d(64)
 96         self.bn2 = nn.BatchNorm1d(128)
		'''



		#self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
		self.conv1 = nn.Conv1d(num_features_in, conv_features_out, 1)
		self.act1 = nn.ReLU()

		#self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(conv_features_out, conv_features_out, 1)
		self.act2 = nn.ReLU()

		#self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(conv_features_out, conv_features_out, 1)
		self.act3 = nn.ReLU()

		#self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
		self.conv4 = nn.Conv1d(conv_features_out, conv_features_out, 1)
		self.act4 = nn.ReLU()

		#self.output1 = nn.Conv2d(feature_size, self.pop_floats, kernel_size=3, padding=1)
		'''
		self.output1 = nn.Conv1d(conv_features_out, self.pop_floats, 1)
		self.output2 = nn.Conv1d(conv_features_out, self.normal_floats * self.normal_max_rows, 1)
		'''
		if pop_floats != -1:
			self.output  = nn.Conv1d(conv_features_out, self.pop_floats, 1)
		else:
			#self.output  = nn.Conv1d(conv_features_out, self.normal_floats * self.normal_max_rows, 1)
			self.output  = nn.Conv1d(conv_features_out, self.normal_floats, 1)

		if self.debug:
			print(f'RegressionModel.__init__() - pop_floats: {pop_floats} - normal_floats: {normal_floats} - normal_max_rows: {normal_max_rows}')

	def forward(self, x):
		# [N, C, W, H]
		# RuntimeError: Given groups=1, weight of size [128, 256, 1], expected input[1, 4, 256] to have 256 channels, but got 4 channels instead
		# RuntimeError: Given groups=1, weight of size [1, 256, 1], expected input[1, 4, 256] to have 256 channels, but got 4 channels instead
		# out is B x C x W x H, with C = 4*num_anchors
		if self.debug:
			print(f'RegressionModel.forward() - {x.shape = }')
		x = x.permute(1, 0)
		if self.debug:
			print(f'RegressionModel.forward() - {x.shape = }')

		out = self.conv1(x)
		out = self.act1(out)

		out = self.conv2(out)
		out = self.act2(out)

		out = self.conv3(out)
		out = self.act3(out)

		out = self.conv4(out)
		out = self.act4(out)

		'''
		out_pop  = self.output1(out)
		out_norm = self.output2(out)
		'''
		out = self.output(out)

		if self.debug:
			print(f'RegressionModel.forward() - {self.name} - out.shape : {out.shape } - out : {out}')
		out = out.permute(1, 0).contiguous()
		if self.debug:
			print(f'RegressionModel.forward() - {self.name} - out.shape : {out.shape } - out : {out}')

		'''
		if self.debug:
			print(f'RegressionModel.forward() - out_pop.shape : {out_pop.shape } - out_pop : {out_pop}')
			print(f'RegressionModel.forward() - out_norm.shape: {out_norm.shape} - out_norm: {out_norm}')
		# out is B x C x W x H, with C = 4*num_anchors
		#out = out.permute(0, 2, 3, 1)
		out_pop  = out_pop.permute (1, 0)
		out_norm = out_norm.permute(1, 0)
		if self.debug:
			print(f'RegressionModel.forward() - out_pop.shape : {out_pop.shape } - out_pop : {out_pop}')
			print(f'RegressionModel.forward() - out_norm.shape: {out_norm.shape} - out_norm: {out_norm}')

		#out = out.contiguous().view(out.shape[0], -1, 3)
		out_pop  = out_pop.contiguous() #.view(out.shape[0], -1, 3)
		out_norm = out_norm.contiguous()
		if self.debug:
			print(f'RegressionModel.forward() - out_pop.shape : {out_pop.shape } - out_pop : {out_pop}')
			print(f'RegressionModel.forward() - out_norm.shape: {out_norm.shape} - out_norm: {out_norm}')

		return [out_pop, out_norm]
		'''
		return out


class CombinedLoss(nn.Module):
	def __init__(self, debug=False):
		super().__init__()
		self.classification_loss = nn.CrossEntropyLoss()
		self.regression_loss = nn.MSELoss()

	def forward(self, cls_pred, cls_target, rot_pred, rot_target, typ_pred, typ_target, pop_pred, pop_target, norm_pred, norm_target):
		cls_loss  = self.classification_loss(cls_pred, cls_target)
		rot_loss  = self.classification_loss(rot_pred, rot_target)
		typ_loss  = self.classification_loss(typ_pred, typ_target)

		pop_loss  = self.regression_loss(pop_pred, pop_target)
		norm_loss = self.regression_loss(norm_pred, norm_target)

		print(f'CombinedLoss.forward() - cls_loss: {cls_loss} - rot_loss: {rot_loss} - typ_loss: {typ_loss} - pop_loss: {pop_loss} - norm_loss: {norm_loss}')

		loss = cls_loss + rot_loss + typ_loss + pop_loss + norm_loss

		return loss

'''''
# YOLOv8
class ExtendedSegment(Segment):
    """Extends the Segment class to add a regression head predicting a 6D vector."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.regression_head = nn.ModuleList(nn.Sequential(
            Conv(x, max(x // 4, 128), 3),
            Conv(max(x // 4, 128), max(x // 4, 128), 3),
            nn.Conv2d(max(x // 4, 128), 6, 1),
            nn.Sigmoid()) for x in ch)  # Produces a 6D vector for each anchor and applies sigmoid activation
'''''





class get_model(nn.Module):
	def __init__(self, out_features=40, normal_channel=True, y_range=None, debug=False):
		super(get_model, self).__init__()
		self.debug = debug

		if normal_channel:
			channel = 6
		else:
			channel = 3

		self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
		'''
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		#self.fc3 = nn.Linear(256, out_features)
		self.dropout = nn.Dropout(p=0.4)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)
		'''
		'''
		self.conv1 = torch.nn.Conv1d(1024, 512, 1)
		self.conv2 = torch.nn.Conv1d(512 , 256, 1)
		self.conv3 = torch.nn.Conv1d(256 , 128, 1)
		self.conv4 = torch.nn.Conv1d(128 ,  64, 1)
		self.bn1   = nn.BatchNorm1d(1024)
		self.bn2   = nn.BatchNorm1d(512)
		self.bn3   = nn.BatchNorm1d(256)
		self.bn4   = nn.BatchNorm1d(128)
		self.dout1 = nn.Dropout(p=0.1),
		self.dout2 = nn.Dropout(p=0.1),
		self.dout3 = nn.Dropout(p=0.1),
		self.dout4 = nn.Dropout(p=0.1),
		'''

		dim = 1024
		self.bnconv1 = nn.Sequential(
			nn.BatchNorm1d(dim),
			#nn.Dropout(p=0.1),
			nn.Linear(dim, dim//2),
			nn.ReLU(inplace=True)
		)
		dim = 512
		self.bnconv2 = nn.Sequential(
			nn.BatchNorm1d(dim),
			#nn.Dropout(p=0.1),
			nn.Linear(dim, dim//2),
			nn.ReLU(inplace=True)
		)
		dim = 256
		self.bnconv3 = nn.Sequential(
			nn.BatchNorm1d(dim),
			#nn.Dropout(p=0.1),
			nn.Linear(dim, dim//2),
			nn.ReLU(inplace=True)
		)
		dim = 128
		self.bnconv4 = nn.Sequential(
			nn.BatchNorm1d(dim),
			#nn.Dropout(p=0.1),
			nn.Linear(dim, dim//2),
			nn.ReLU(inplace=True)
		)

		self.pop_floats	     =  3	# point on plane coordinates (x,y,z)
		self.normal_floats   =  3	# normal (nx,ny,nz)
		self.normal_max_rows = 14	# 14x normals per each input point cloud

		#self.regr = RegressionHead(pop_floats=3, normal_floats=3, normal_max_rows=14, debug=False)
		#self.regr = RegressionModel(num_features_in=256, conv_features_out=128, pop_floats=3, normal_floats=3, normal_max_rows=14, debug=True)
		#self.regr = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=3, normal_floats=3, normal_max_rows=14, debug=True)
		#self.regr = [RegressionModel(num_features_in=256, conv_features_out=32, debug=True) for i in range(self.pop_floats)]
		'''
		self.regr_pop  = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=3 , normal_floats=-1, normal_max_rows=-1, debug=False)
		self.regr_norm = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=-1, normal_floats=3 , normal_max_rows=14, debug=False)
		'''
		#self.regr_pop  = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=3 , normal_floats=-1, debug=False)
		#self.regr_pop  = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=1 , normal_floats=-1, debug=False)
		#self.regr_pop  = nn.ModuleList([MLPRegressionHead(in_dim=256, hidden_dim=256, pop_floats=1, normal_floats=-1, normal_max_rows=-1, debug=False) for i in range(self.pop_floats)])
		#self.regr_pop  = MLPRegressionHead(in_dim=256, hidden_dim=256, pop_floats=3, normal_floats=-1, normal_max_rows=-1, debug=False)
		#self.regr_pop  = MLPRegressionHead(in_dim=64, hidden_dim=256, pop_floats=3, normal_floats=-1, normal_max_rows=-1, debug=False)
		self.regr_pop  = nn.ModuleList([MLPRegressionHead(in_dim=64, pop_floats=1, normal_floats=-1, normal_max_rows=-1, debug=False) for i in range(self.pop_floats)])

		#self.regr_norm = [RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=-1, normal_floats=3 , debug=True) for i in range(self.normal_max_rows)]
		#self.regr_norm = nn.ModuleList([RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=-1, normal_floats=3 , debug=False) for i in range(self.normal_max_rows)])
		#self.regr_norm = MLPRegressionHead(in_dim=256, hidden_dim=256, pop_floats=-1, normal_floats=3, normal_max_rows=14, debug=False)


		self.sigmoid_range = SigmoidRange(0, 1)

		'''
		self.y_range = y_range

		if self.y_range is not None:
			self.sigmoid_range = SigmoidRange(self.y_range[0], self.y_range[1])
		else:
			self.relu = nn.ReLU()
		'''

	def forward(self, x):
		if self.debug:
			print(f'get_model.forward() {x.shape = }')
		x, trans, trans_feat = self.feat(x)
		'''
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))
		'''
		x = self.bnconv1(x)
		x = self.bnconv2(x)
		x = self.bnconv3(x)
		x = self.bnconv4(x)

		if self.debug:
			print(f'get_model.forward() {x.shape = }')
		#x = self.regr(x)
		#x = [self.regr[i](x) for i in range(self.pop_floats)]
		#x_pop, x_norm = self.regr_pop(x), self.regr_norm(x)
		#x_pop  = self.regr_pop(x)
		#x_norm = self.regr_norm(x)

		x_pop = []
		for head in self.regr_pop:
			x_pop.append(head(x))

		'''
		for idx in range(self.pop_floats):
			x_pop.append(self.regr_pop[idx](x))
		'''
		x_pop = torch.cat(x_pop, dim=1)

		if self.debug:
			print(f'get_model.forward() - torch.cat - x_pop: {type(x_pop)} - x_pop: {x_pop}')

		x_pop = self.sigmoid_range(x_pop)

		if self.debug:
			print(f'get_model.forward() - sigmoid   - x_pop: {type(x_pop)} - x_pop: {x_pop}')
		'''
		x_norm = []
		for idx in range(self.normal_max_rows):
			x_norm.append(self.regr_norm[idx](x))
		'''
		'''
		x = self.fc3(x)
		if self.y_range is not None:
			x = self.sigmoid_range(x)
		else:
			x = F.log_softmax(x, dim=1)
		'''


		if self.debug:
			#print(f'get_model.forward() - Returning {type(x_pop)} - {type(x_norm)}')
			print(f'get_model.forward() - Returning {type(x_pop)}')

		#return x, trans_feat
		#return [x_pop, x_norm], trans_feat
		return [x_pop, None], trans_feat

	def list_target_to_cuda_float_tensor(self, target, cuda=True, debug=False):
		for idx,tgt in enumerate(target):		# because now target is a list of lists/np.arrays
			if debug:
				print(f'cpu  target[{idx}]: {type(tgt)} -  {tgt}')
			target[idx] = torch.tensor(tgt)
			if cuda:
				target[idx] = target[idx].cuda()
			if debug:
				print(f'cuda target[{idx}]: {type(target[idx])} -  {target[idx]}')

class get_loss(torch.nn.Module):
	def __init__(self, y_range=None, mat_diff_loss_scale=0.000001, dataset='symmetry', debug=False):
		super(get_loss, self).__init__()
		self.mat_diff_loss_scale = mat_diff_loss_scale
		self.y_range = y_range
		self.dataset = dataset
		self.debug   = debug

		self.batch_counter = 0

	def forward(self, pred, target, trans_feat):

		self.batch_counter += 1


		#print(f'get_loss.forward() - self.debug: {self.debug} - self.batch_counter: {self.batch_counter}')
		if self.debug and self.batch_counter % 100 == 0:
			print(f'get_loss.forward() - len(pred): {len(pred)} - len(target): {len(target)}')
			print(f'get_loss.forward() - pred[0].shape: {pred[0].shape} - target[0].shape: {target[0].shape}')

		pr  = pred[0][0]
		tgt = target[0][0]#.reshape(pr.shape)

		if self.debug and self.batch_counter % 100 == 0:
			print(f'get_loss.forward() - pr.shape : {pr.shape}')
			print(f'get_loss.forward() - tgt.shape: {tgt.shape}')

		#loss= F.mse_loss(pr, tgt.float())
		loss = (pr-tgt).abs().sum()			# not even a MAE/L1Loss

		'''
		loss = 0
		for jdx,itm in enumerate(zip(pr, tgt)):
			pr_itm  = itm[0]
			tgt_itm = itm[1]
			loss    = F.mse_loss(itm[0], itm[1].float()) + loss
			break
		'''

		#loss = (pred[0] - target[0]) ** 2
		#loss = F.mse_loss(pred[0], target[0].float())
		if self.debug and self.batch_counter % 100 == 0:
			print(f'get_loss.forward() - pred[0].shape: {pred[0].shape}')
			print(f'get_loss.forward() - target[0].shape: {target[0].shape}')
			print(f'get_loss.forward() - loss.shape: {loss.shape}')
			print(f'get_loss.forward() - pred[0]: {pred[0]}')
			print(f'get_loss.forward() - target[0]: {target[0]}')
			print(f'get_loss.forward() - loss: {loss}')

		mat_diff_loss = feature_transform_regularizer(trans_feat)

		if self.debug and self.batch_counter % 100 == 0:
			print(f'get_loss.forward() - mat_diff_loss: {mat_diff_loss}')

		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale

		if self.debug and self.batch_counter % 100 == 0:
			print(f'get_loss.forward() - total_loss: {total_loss}')

		return total_loss, loss, mat_diff_loss * self.mat_diff_loss_scale
		#return loss


		if self.dataset == 'symmetry':
			torch.set_printoptions(profile="full")
			torch.set_printoptions(linewidth=210)
			if self.debug:
				print(f'get_loss.forward() - type(pred): {type(pred)} - type(target): {type(target)}')
				print(f'get_loss.forward() - len(pred): {len(pred)} - len(target): {len(target)}')
				print(f'get_loss.forward() - pred: {pred} - target: {target}')
			torch.set_printoptions(profile="default")
			if self.debug:
				if isinstance(pred, torch.Tensor):
					print(f'get_loss.forward() - pred.shape: {pred.shape}')
				if isinstance(target, list):
					for idx,tgt in enumerate(target):
						if isinstance(tgt, torch.Tensor):
							print(f'get_loss.forward() - tensor target[{idx}].shape: {tgt.shape}')
						elif isinstance(tgt, l):
							print(f'get_loss.forward() - list   target[{idx}].len  : {len(tgt)}')
				print(f'get_loss.forward() - pred: {pred} - target: {target}')
			#pred = pred.squeeze(1)
			#loss = F.mse_loss(pred, target)

			'''
			for idx,pr in enumerate(pred):
				tgt = target[idx].reshape(pr.shape)
				loss_itm = F.mse_loss(pr, tgt.float())
				print(f'get_loss.forward() - loss_itm[{idx}]: {loss_itm}')
				loss_lst.append(loss_itm)
			'''
			loss = self.list_target_loss_impl(pred, target)
		else:
			loss = F.nll_loss(pred, target)
		mat_diff_loss = feature_transform_regularizer(trans_feat)
		if self.debug:
			print(f'get_loss.forward() - final loss: {loss} - mat_diff_loss: {mat_diff_loss}')

		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
		#total_loss = loss_lst[0]

		if self.debug:
			print(f'get_loss.forward() - total_loss: {total_loss} - type(total_loss): {type(total_loss)} - dtype(total_loss): {total_loss.dtype}')
		return total_loss.float()

	@staticmethod
	def list_target_loss_impl(pred, target, debug=False):
		torch.set_printoptions(precision=3)
		torch.set_printoptions(sci_mode=False)

		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')
		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')
		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')

		loss = 0
		#loss_lst = []
		for idx,pr in enumerate(pred):
			print(f'list_target_loss_impl() ==================================================================================')
			print(f'list_target_loss_impl() ==================================================================================')
			print(f'list_target_loss_impl() ==================================================================================')
			if isinstance(pr, torch.Tensor):		# this is model.regr_pop, a predictor of 3 floats = [bs, 3] = [5, 3]
				if debug:
					print(f'Taking into consideration (and reshaping) pred[{idx}]:\n{pr} with shape {pr.shape}\nand target[{idx}]:\n{target[idx]} with shape {target[idx].shape}')
				tgt = target[idx].reshape(pr.shape)
			elif isinstance(pr, list):			# this is model.regr_norm, a predictor of 3x14 floats = [bs, 3, 14] = [5, 3, 14]
				if False:
					print(f'{pr[0].shape = }')
				print(f'Taking into consideration (and reshaping) pred[{idx}]:\n{pr} with len {len(pr)}\nand target[{idx}]:\n{target[idx]} with shape {target[idx].shape}')
				#tgt = target[idx].permute(0, 2, 1).reshape([len(pr), pr[0].shape[0], pr[0].shape[1]])			# here we want [5, 3, 14] - [bs, normal_floats, normal_max_rows]
				tgt = [target[idx][:,:,i] for i in range(target[idx].shape[2])]
			if debug:
				print(f'list_target_loss_impl() - reshaped target[{idx}]: {tgt.shape} - type(tgt): {type(tgt)} - len(tgt): {len(tgt)} - {tgt}')
			#part_loss_lst = []
			if debug:
				print(f'list_target_loss_impl() - looping over pr and tgt: {len(pr)} - {len(tgt)} - with shapes: {pr[0].shape} - {tgt[0].shape}')
				print(f'list_target_loss_impl() - looping over\n{pr}\n ------------------------ and\n{tgt}')
			for jdx,itm in enumerate(zip(pr, tgt)):
				pr_itm   = itm[0]
				tgt_itm  = itm[1]
				print(f'list_target_loss_impl()\n[{idx}][{jdx}] - {pr_itm.shape} - {tgt_itm.shape} - pr_itm: {pr_itm} - tgt_itm: {tgt_itm}')
				if len(pr_itm.shape) == 1:
					loss_itm = F.mse_loss(itm[0], itm[1].float())
				else:
					for kdx,sub_itm in enumerate(zip(pr_itm, tgt_itm)):
						pr_sub_itm   = sub_itm[0]
						tgt_sub_itm  = sub_itm[1]
						print(f'list_target_loss_impl()\n[{idx}][{jdx}][{kdx}] - {pr_sub_itm.shape} - {tgt_sub_itm.shape} - pr_sub_itm: {pr_sub_itm} - tgt_sub_itm: {tgt_sub_itm}')
						loss_subitm = F.mse_loss(pr_sub_itm, tgt_sub_itm.float())
						print(f'list_target_loss_impl() - loss_subitm[{idx}][{jdx}][{kdx}]: {loss_subitm}')
					loss_itm = loss_subitm + loss_itm
				#loss_itm = torch.sum(torch.abs(itm[0], itm[1].float()))
				if debug or True:
					print(f'list_target_loss_impl() - loss_itm[{idx}][{jdx}]: {loss_itm}')
				#part_loss_lst.append(loss_itm)
				#loss_lst.append(loss_itm)
				loss = loss_itm + loss
			#loss_lst.append(sum(part_loss_lst))
			if debug:
				print(f'list_target_loss_impl() - {loss = }')
			print(f'list_target_loss_impl() ##################################################################################')
			print(f'list_target_loss_impl() ##################################################################################')
			print(f'list_target_loss_impl() ##################################################################################')
		#loss = sum(loss_lst)
		'''
		if debug:
			print(f'list_target_loss_impl() - loss_lst: {loss_lst}')
		'''

		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {loss = }')
		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		return loss#, loss_lst
