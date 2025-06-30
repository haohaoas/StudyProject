class Solution:
    def colorTheGrid(self, m: int, n: int) -> int:
        MOD = 10**9 + 7
        def generate():
            states = []
            def dfs(path):
                if len(path) == m:
                    states.append(tuple(path))
                    return
                for c in [0, 1, 2]:
                    if not path or path[-1] != c:
                        dfs(path + [c])
            dfs([])
            return states

        states = generate()
        total_states = len(states)

        graph = [[] for _ in range(total_states)]
        for i, s in enumerate(states):
            for j, t in enumerate(states):
                if all(sc != tc for sc, tc in zip(s, t)):
                    graph[i].append(j)

        dp = [1] * total_states

        for _ in range(1, n):
            new_dp = [0] * total_states
            for i in range(total_states):
                for j in graph[i]:
                    new_dp[i] = (new_dp[i] + dp[j]) % MOD
            dp = new_dp

        return sum(dp) % MOD


        

if __name__ == '__main__':
    print(Solution().colorTheGrid(m = 1, n = 1))
    print(Solution().colorTheGrid(m = 1, n = 2))
    print(Solution().colorTheGrid(m = 5, n = 5))