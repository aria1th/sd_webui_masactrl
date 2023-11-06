
from typing import Tuple
import torch.nn.functional as F

def find_original_values(w_scaled:int, h_scaled:int, mult_original:int) -> Tuple[int, int]:
    """
    Find the original height and width.
    
    w_scaled = int(width * scale)
    h_scaled = int(height * scale)
    mult_original = int(width * height)
    """
    a, b = min(w_scaled, h_scaled), max(w_scaled, h_scaled) 
    is_width_smaller = w_scaled == a
    # get the decomposed factors of mult_original
    for factor in range(2, int(mult_original ** 0.5 + 1)):
        if mult_original % factor == 0:
            factor, another = factor, mult_original // factor
            # try to get the scale
            scale_a, scale_b = a /factor, b / another
            # if scales are close enough
            if abs(scale_a - scale_b) < 0.1:
                return (factor, another) if is_width_smaller else (another, factor)
            
    return (0, 0)

def functional_reshape_values(q, k, v, mask):
    # mask based on height and width but depends on the original height and width
    # [1, 8, 384, 160] , [1, 8, 551, 160], [1, 8, 551, 160], [1, 8, 384, 384] or None
    # 24, 16 -> 29, 19 (scale 1.2)
    # get target k/v shape
    shape_2d = k.shape[2] #551 = 29 * 19
    shape_2d_orig = q.shape[2] #384 = 24 * 16
    if shape_2d_orig == shape_2d:
        return q, k, v, mask
    # reshape 551 to 384
    interpolate_k = F.interpolate(k, size=(shape_2d_orig, k.shape[3]), mode='nearest')
    # softmax
    softmax_k = F.softmax(interpolate_k, dim=2)
    # reshape 551 to 384
    interpolate_v = F.interpolate(v, size=(shape_2d_orig, v.shape[3]), mode='nearest')
    softmax_v = F.softmax(interpolate_v, dim=2)
    # return
    return q, softmax_k, softmax_v, mask