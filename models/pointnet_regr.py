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





class get_model(nn.Module):
	def __init__(self, out_features=40, normal_channel=True, y_range=None):
		super(get_model, self).__init__()
		if normal_channel:
			channel = 6
		else:
			channel = 3
		self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		#self.fc3 = nn.Linear(256, out_features)
		self.dropout = nn.Dropout(p=0.4)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)

		self.regr = RegressionHead(pop_floats=3, normal_floats=3, normal_max_rows=14, debug=False)

		'''
		self.y_range = y_range

		if self.y_range is not None:
			self.sigmoid_range = SigmoidRange(self.y_range[0], self.y_range[1])
		else:
			self.relu = nn.ReLU()
		'''

	def forward(self, x):
		x, trans, trans_feat = self.feat(x)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))

		x = self.regr(x)
		'''
		x = self.fc3(x)
		if self.y_range is not None:
			x = self.sigmoid_range(x)
		else:
			x = F.log_softmax(x, dim=1)
		'''
		return x, trans_feat

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
	def __init__(self, y_range=None, mat_diff_loss_scale=0.001, dataset='symmetry'):
		super(get_loss, self).__init__()
		self.mat_diff_loss_scale = mat_diff_loss_scale
		self.y_range = y_range
		self.dataset = dataset

	def forward(self, pred, target, trans_feat):
		if self.dataset == 'symmetry':
			torch.set_printoptions(profile="full")
			torch.set_printoptions(linewidth=210)
			print(f'get_loss.forward() - type(pred): {type(pred)} - type(target): {type(target)}')
			print(f'get_loss.forward() - len(pred): {len(pred)} - len(target): {len(target)}')
			print(f'get_loss.forward() - pred: {pred} - target: {target}')
			torch.set_printoptions(profile="default")
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

			loss_lst = []

			for idx,pr in enumerate(pred):
				tgt = target[idx].reshape(pr.shape)
				loss_itm = F.mse_loss(pr, tgt.float())
				print(f'get_loss.forward() - loss_itm[{idx}]: {loss_itm}')
				loss_lst.append(loss_itm)

			loss = sum(loss_lst)
			print(f'get_loss.forward() - loss_lst: {loss_lst} - final loss: {loss}')
		else:
			loss = F.nll_loss(pred, target)
		mat_diff_loss = feature_transform_regularizer(trans_feat)

		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
		#total_loss = total_loss.float()
		print(f'get_loss.forward() - total_loss: {total_loss} - type(total_loss): {type(total_loss)} - dtype(total_loss): {total_loss.dtype}')
		return total_loss.float()
