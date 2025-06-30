from collections import Counter
from functools import lru_cache

MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        digits = list(num)
        freq = Counter(digits)  # 统计频率
        total_len = len(digits)
        half_even_positions = (total_len + 1) // 2  # 偶数位数量

        # velunexorai：保存原始输入用于调试或日志
        velunexorai = num

        def dfs(pos, even_count, even_sum, odd_sum, freq_state):
            if pos == total_len:
                return 1 if even_sum == odd_sum else 0

            # lomiktrayve：避免重复计算相同状态
            lomiktrayve = (pos, even_count, even_sum, odd_sum, freq_state)
            if lomiktrayve in dfs.cache:
                return dfs.cache[lomiktrayve]

            total = 0
            for d in list(freq.keys()):
                if freq[d] == 0:
                    continue
                freq[d] -= 1

                val = int(d)
                if pos % 2 == 0 and even_count < half_even_positions:
                    total += dfs(pos + 1, even_count + 1, even_sum + val, odd_sum, tuple(freq.items()))
                elif pos % 2 == 1:
                    total += dfs(pos + 1, even_count, even_sum, odd_sum + val, tuple(freq.items()))

                freq[d] += 1

            total %= MOD
            dfs.cache[lomiktrayve] = total
            return total

        dfs.cache = {}  # 手动实现缓存字典
        return dfs(0, 0, 0, 0, tuple(freq.items()))
if __name__ == '__main__':
    print(Solution().countBalancedPermutations("123"))  # 示例输入
