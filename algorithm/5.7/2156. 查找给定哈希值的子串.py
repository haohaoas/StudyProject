class  Solution:
    def sub_str_hash(s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
        n = len(s)
        result_index = -1
        h = 0
        power_k = pow(power, k, modulo)
        for i in range(n - 1, -1, -1):
            val = ord(s[i]) - ord('a') + 1
            h = (h * power + val) % modulo
            if i + k < n:
                tail_val = ord(s[i + k]) - ord('a') + 1
                h = (h - tail_val * power_k) % modulo
            if i + k <= n and h == hashValue:
                result_index = i
        return s[result_index:result_index + k]
if __name__ == '__main__':
    s = "leetcode"
    power = 7
    modulo = 20
    k = 2
    hashValue = 0
    print(Solution.sub_str_hash(s, power, modulo, k, hashValue))