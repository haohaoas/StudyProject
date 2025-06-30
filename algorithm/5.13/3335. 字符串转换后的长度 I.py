class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:

        MOD = 10 ** 9 + 7

        counts = [0] * 26
        for char in s:
            counts[ord(char) - ord('a')] += 1
        for _ in range(t):
            new_counts = [0] * 26
            for i in range(26):
                if i == 25:
                    new_counts[0] = (new_counts[0] + counts[i]) % MOD
                    new_counts[1] = (new_counts[1] + counts[i]) % MOD
                else:
                    new_counts[(i + 1) % 26] = (new_counts[(i + 1) % 26] + counts[i]) % MOD
            counts = new_counts

        return sum(counts) % MOD


if __name__ == '__main__':
    s ="jqktcurgdvlibczdsvnsg"
    t = 7517
    print(Solution().lengthAfterTransformations(s, t))