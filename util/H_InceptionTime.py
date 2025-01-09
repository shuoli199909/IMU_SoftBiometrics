import torch
import torch.nn as nn
import numpy as np

"""
Unofficial InceptionTime Pytorch implementation: https://github.com/TheMrGhostman/InceptionTime-Pytorch/tree/master.
A parameter for the hybrid case is added to the architecture.
"""

def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes

def pass_through(X):
	return X

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    

class Inception(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[10, 20, 40], bottleneck_channels=32, activation=nn.ReLU(), use_hybrid=False, return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if number of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
  		"""
		super(Inception, self).__init__()
		self.return_indices=return_indices
		self.use_hybrid = use_hybrid
  
		if in_channels > 1:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = 1

		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding='same', 
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding='same', 
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding='same', 
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=n_filters, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
  
		if use_hybrid:
			self.hybrid = HybridLayer(input_channels=bottleneck_channels)
			self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters+17)
		else:
			self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		Z4 = self.conv_from_maxpool(Z_maxpool)
		# step 3
		if self.use_hybrid:
			Z5 = self.hybrid(Z_bottleneck)
			Z = torch.cat([Z1, Z2, Z3, Z4, Z5], dim=1)
		else:
			Z = torch.cat([Z1, Z2, Z3, Z4], dim=1)
		
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z


class InceptionBlock(nn.Module):   
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[10,20,40], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), use_hybrid=False, return_indices=False):		
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.return_indices = return_indices
		self.activation = activation
  
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation, use_hybrid=use_hybrid,
							return_indices=return_indices
							)
		self.inception_2 = Inception(
							in_channels=4*n_filters if not use_hybrid else 4*n_filters+17,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_3 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters,
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=4*n_filters
									)
								)

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
			Z, i2 = self.inception_2(Z)
			Z, i3 = self.inception_3(Z)
		else:
			Z = self.inception_1(X)
			Z = self.inception_2(Z)
			Z = self.inception_3(Z)
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		if self.return_indices:
			return Z,[i1, i2, i3]
		else:
			return Z


class InceptionTranspose(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		"""
		super(InceptionTranspose, self).__init__()
		self.activation = activation
		self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.conv_to_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
		self.bottleneck = nn.Conv1d(
								in_channels=3*bottleneck_channels, 
								out_channels=out_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

	def forward(self, X, indices):
		Z1 = self.conv_to_bottleneck_1(X)
		Z2 = self.conv_to_bottleneck_2(X)
		Z3 = self.conv_to_bottleneck_3(X)
		Z4 = self.conv_to_maxpool(X)

		Z = torch.cat([Z1, Z2, Z3], dim=1)
		MUP = self.max_unpool(Z4, indices)
		BN = self.bottleneck(Z)
		# another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution
		
		return self.activation(self.batch_norm(BN + MUP))


class InceptionTransposeBlock(nn.Module):
	def __init__(self, in_channels, out_channels=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU()):
		super(InceptionTransposeBlock, self).__init__()
		self.use_residual = use_residual
		self.activation = activation
		self.inception_1 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_2 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_3 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=out_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.ConvTranspose1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=out_channels
									)
								)

	def forward(self, X, indices):
		assert len(indices)==3
		Z = self.inception_1(X, indices[2])
		Z = self.inception_2(Z, indices[1])
		Z = self.inception_3(Z, indices[0])
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		return Z


"""
New part: unofficial Hybrid InceptionTime Pytorch implementation.
Official Hybrid InceptionTime tensorflow implementation: https://github.com/MSD-IRIMAS/CF-4-TSC/tree/main
"""

class HybridLayer(nn.Module):
    def __init__(self, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64], activation=nn.ReLU()):
        super(HybridLayer, self).__init__()
        self.input_channels = input_channels
        self.kernel_sizes = kernel_sizes
        self.conv_layers = self._create_conv_layers() 
        self.activation = activation  

    def _create_conv_layers(self):
        layers = []
        for kernel_size in self.kernel_sizes:
            # Increasing detection filter
            filter_inc = self._create_filter(kernel_size, pattern='increase')
            conv_inc = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size,
                                 padding='same', bias=False)
            conv_inc.weight.data = filter_inc
            conv_inc.weight.requires_grad = False
            layers.append(conv_inc)

            # Decreasing detection filter
            filter_dec = self._create_filter(kernel_size, pattern='decrease')
            conv_dec = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size,
                                 padding='same', bias=False)
            conv_dec.weight.data = filter_dec
            conv_dec.weight.requires_grad = False
            layers.append(conv_dec)

            # Peak detection filter, excluding the smallest kernel size for symmetry
            if kernel_size > 2:
                filter_peak = self._create_filter(kernel_size, pattern='peak')
                conv_peak = nn.Conv1d(in_channels=self.input_channels, out_channels=1, kernel_size=kernel_size + kernel_size // 2,
                                      padding='same', bias=False)
                conv_peak.weight.data = filter_peak
                conv_peak.weight.requires_grad = False                
                layers.append(conv_peak)
        return nn.ModuleList(layers)

    def _create_filter(self, k, pattern):
        if pattern == 'increase':
            filter_ = np.ones((1, self.input_channels, k))
            filter_[:, :, np.arange(k) % 2 == 0] = -1
        elif pattern == 'decrease':
            filter_ = np.ones((1, self.input_channels, k))
            filter_[:, :, np.arange(k) % 2 != 0] = -1
        elif pattern == 'peak':
            filter_ = np.zeros((1, self.input_channels, k+k//2))
            
            xmesh = np.linspace(start=0, stop=1, num=k // 4 + 1)[1:]
            
            filter_left = xmesh ** 2
            filter_right = np.flip(filter_left)
            
            filter_[:, :, 0:k//4] = -filter_left
            filter_[:, :, k//4:k//2] = -filter_right
            filter_[:, :, k//2:3*k//4] = 2 * filter_left
            filter_[:, :, 3*k//4:k] = 2 * filter_right
            filter_[:, :, k:5*k//4] = -filter_left
            filter_[:, :, 5*k//4:] = -filter_right
        return torch.tensor(filter_, dtype=torch.float32)

    def forward(self, x):
        outputs = []
        for conv in self.conv_layers:
            outputs.append(conv(x))
        outputs = torch.cat(outputs, dim=1)
        return self.activation(outputs)


class HybridInceptionTime(nn.Module):
    def __init__(self, in_channels, number_classes, kernel_sizes=[10, 20, 40], n_filters=32, bottleneck_channels=32):
        super(HybridInceptionTime, self).__init__()
        self.inception_block1 = InceptionBlock(
			in_channels=in_channels,
			n_filters=n_filters,
			kernel_sizes=kernel_sizes,
   			bottleneck_channels=bottleneck_channels,
			use_residual=True,
			activation=nn.ReLU(),
			use_hybrid=True
		)
        self.inception_block2 = InceptionBlock(
			in_channels=n_filters*4,
			n_filters=n_filters,
   			kernel_sizes=kernel_sizes,
			bottleneck_channels=bottleneck_channels,
			use_residual=True,
			activation=nn.ReLU(),
   			use_hybrid=False
		)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = Flatten(out_features=n_filters*4*1)
        self.fc = nn.Linear(in_features=n_filters*4*1, out_features=number_classes)
        
    def forward(self, X):
        X = self.inception_block1(X)
        X = self.inception_block2(X)
        X = self.avg_pool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X