import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        self.Q = Q
        self.K = K
        self.V = V
        d_k = Q.shape[-1]
        K_shape = [i for i in range(len(K.shape))]
        temp = K_shape[-1]
        K_shape[-1] = K_shape[-2]
        K_shape[-2] = temp
        scaled_dot_product = (Q@np.transpose(K, K_shape))/np.sqrt(d_k)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product = scaled_dot_product * (1 - mask) - self.eps * mask

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores@V

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions  
        attention_shape = [i for i in range(len(self.attention_scores.shape))]
        temp = attention_shape[-1]
        attention_shape[-1] = attention_shape[-2]
        attention_shape[-2] = temp 
        d_V = self.attention_scores.transpose(attention_shape)@d_output
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        V_shape = [i for i in range(len(self.V.shape))]
        temp = V_shape[-1]
        V_shape[-1] = V_shape[-2]
        V_shape[-2] = temp
        d_attention_scores = d_output@self.V.transpose(V_shape)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.Q.shape[-1])
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product@self.K
        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        scale_shape = [i for i in range(len(d_scaled_dot_product.shape))]
        temp = scale_shape[-1]
        scale_shape[-1] = scale_shape[-2]
        scale_shape[-2] = temp
        d_K = d_scaled_dot_product.transpose(scale_shape)@self.Q
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

