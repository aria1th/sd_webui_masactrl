from functools import lru_cache


@lru_cache(maxsize=128)
def get_closest_defactorization(factors_1: int, factors_2: int, target_num: int):
    """
    Calculate closest defactorization of target_num, based on factors_1 and factors_2.
        @param factors_1, factors_2: expected factors
        @param target_num: target number
    
        @return: [factor_1, factor_2] : factor_1 * factor_2 = target_num considering the closest defactorization
        
    @example:
        get_closest_defactorization(16, 24, 551) # result is [19, 29] and hidden factor (multiplied) was 1.2
    """
    # check if target_num is a multiple of factors_1 and factors_2
    if target_num == factors_1 * factors_2:
        return [factors_1, factors_2]
    
    # Calculate the ratio
    r = (target_num / (factors_1 * factors_2)) ** 0.5
    
    # Estimate potential factors
    estimated_factor1 = int(factors_1 * r)
    estimated_factor2 = int(factors_2 * r)
    
    # Search range
    search_range = 10
    
    # Check both increasing and decreasing from the estimated factors
    for i in range(search_range):
        factor_a_up = estimated_factor1 + i
        factor_a_down = estimated_factor1 - i
        
        # Check if factor_a_up is a factor of target_num
        if target_num % factor_a_up == 0:
            return [factor_a_up, target_num // factor_a_up]
        
        # Check if factor_a_down is a factor of target_num
        if factor_a_down > 0 and target_num % factor_a_down == 0:
            return [factor_a_down, target_num // factor_a_down]
    
    # If no factors found, return the estimated factors
    return [estimated_factor1, estimated_factor2]
