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
# YOLOv8 regression head
class ExtendedSegment(Segment):
    """Extends the Segment class to add a regression head predicting a 6D vector."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.regression_head = nn.ModuleList(nn.Sequential(
            Conv(x, max(x // 4, 128), 3),
            Conv(max(x // 4, 128), max(x // 4, 128), 3),
            nn.Conv2d(max(x // 4, 128), 6, 1),
            nn.Sigmoid()) for x in ch)  # Produces a 6D vector for each anchor and applies sigmoid activation
'''

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

class RegressionModel(nn.Module):
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

		self.conv1 = nn.Conv1d(num_features_in, conv_features_out, kernel_size=1)
		self.act1 = nn.ReLU()

		self.conv2 = nn.Conv1d(conv_features_out, conv_features_out, kernel_size=1)
		self.act2 = nn.ReLU()

		self.conv3 = nn.Conv1d(conv_features_out, conv_features_out, kernel_size=1)
		self.act3 = nn.ReLU()

		self.conv4 = nn.Conv1d(conv_features_out, conv_features_out, kernel_size=1)
		self.act4 = nn.ReLU()

		#self.output1 = nn.Conv2d(feature_size, self.pop_floats, kernel_size=3, padding=1)
		'''
		self.output1 = nn.Conv1d(conv_features_out, self.pop_floats, 1)
		self.output2 = nn.Conv1d(conv_features_out, self.normal_floats * self.normal_max_rows, 1)
		'''
		if pop_floats != -1:
			self.output  = nn.Conv1d(conv_features_out, self.pop_floats, kernel_size=1)
		else:
			#self.output  = nn.Conv1d(conv_features_out, self.normal_floats * self.normal_max_rows, kernel_size=1)
			self.output  = nn.Conv1d(conv_features_out, self.normal_floats, kernel_size=1)

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


class get_model(nn.Module):
	def __init__(self, out_features=40, normal_channel=True, y_range=None, debug=False):
		super(get_model, self).__init__()
		self.debug = debug

		if normal_channel:
			channel = 6
		else:
			channel = 3

		self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.dropout = nn.Dropout(p=0.4)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)


		self.pop_floats	     =  3	# point on plane coordinates (x,y,z)
		self.normal_floats   =  3	# normal (nx,ny,nz)
		self.normal_max_rows = 14	# 14x normals per each input point cloud

		self.regr_pop  = RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=3 , normal_floats=-1, debug=False)
		#self.regr_norm = [RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=-1, normal_floats=3 , debug=True) for i in range(self.normal_max_rows)]
		self.regr_norm = nn.ModuleList([RegressionModel(num_features_in=256, conv_features_out=32, pop_floats=-1, normal_floats=3 , debug=False) for i in range(self.normal_max_rows)])

	def forward(self, x):
		x, trans, trans_feat = self.feat(x)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))

		if self.debug:
			print(f'get_model.forward() {x.shape = }')

		#x = [self.regr[i](x) for i in range(self.pop_floats)]
		#x_pop, x_norm = self.regr_pop(x), self.regr_norm(x)
		x_pop  = self.regr_pop(x) 
		x_norm = []
		for idx in range(self.normal_max_rows):
			x_norm.append(self.regr_norm[idx](x))

		if self.debug:
			print(f'get_model.forward() - Returning {type(x_pop)} - {type(x_norm)}')

		return [x_pop, x_norm], trans_feat

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
	def __init__(self, y_range=None, mat_diff_loss_scale=0.00001, dataset='symmetry', debug=False):
		super(get_loss, self).__init__()
		self.mat_diff_loss_scale = mat_diff_loss_scale
		self.y_range = y_range
		self.dataset = dataset
		self.debug   = debug

	def forward(self, pred, target, trans_feat):

		pr  = pred[0][0]
		tgt = target[0][0]#.reshape(pr.shape)
		loss= F.mse_loss(pr, tgt.float())

		print(f'get_loss.forward() - pred[0].shape: {pred[0].shape}')
		print(f'get_loss.forward() - target[0].shape: {target[0].shape}')
		print(f'get_loss.forward() - loss.shape: {loss.shape}')
		print(f'get_loss.forward() - pred[0]: {pred[0]}')
		print(f'get_loss.forward() - target[0]: {target[0]}')
		print(f'get_loss.forward() - loss: {loss}')
		mat_diff_loss = feature_transform_regularizer(trans_feat)
		print(f'get_loss.forward() - mat_diff_loss: {mat_diff_loss}')
		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
		print(f'get_loss.forward() - total_loss: {total_loss}')
		return total_loss

		'''
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
			loss = self.list_target_loss_impl(pred, target)
		else:
			loss = F.nll_loss(pred, target)
		mat_diff_loss = feature_transform_regularizer(trans_feat)
		if self.debug:
			print(f'get_loss.forward() - final loss: {loss} - mat_diff_loss: {mat_diff_loss}')

		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale

		if self.debug:
			print(f'get_loss.forward() - total_loss: {total_loss} - type(total_loss): {type(total_loss)} - dtype(total_loss): {total_loss.dtype}')
		return total_loss.float()
		'''

	@staticmethod
	def list_target_loss_impl(pred, target, debug=False):
		torch.set_printoptions(precision=3)
		torch.set_printoptions(sci_mode=False)

		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')
		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')
		print(f'list_target_loss_impl() ----------------------------------------------------------------------------------')

		loss = 0
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
				if debug or True:
					print(f'list_target_loss_impl() - loss_itm[{idx}][{jdx}]: {loss_itm}')
				loss = loss_itm + loss
			if debug:
				print(f'list_target_loss_impl() - {loss = }')
			print(f'list_target_loss_impl() ##################################################################################')
			print(f'list_target_loss_impl() ##################################################################################')
			print(f'list_target_loss_impl() ##################################################################################')

		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {loss = }')
		print(f'list_target_loss_impl() ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
		return loss
