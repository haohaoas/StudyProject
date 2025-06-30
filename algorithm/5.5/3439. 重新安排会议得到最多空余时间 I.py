from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        durations = [endTime[i] - startTime[i] for i in range(n)]# 会议的持续时间
        max_gap = 0
        total_duration = sum(durations[:k])# 前k个会议的总时长
        for i in range(n - k + 1):
            window_start = endTime[i - 1] if i > 0 else 0
            window_end = startTime[i + k] if i + k < n else eventTime
            free_space = window_end - window_start - total_duration
            if free_space > max_gap:
                max_gap = free_space
            if i + k < n:
                total_duration += durations[i + k] - durations[i]
        for i in range(n - 1):
            gap = startTime[i + 1] - endTime[i]
            if gap > max_gap:
                max_gap = gap
        return max_gap



if __name__ == '__main__':
    eventTime = 10
    k = 1
    startTime = [0,2,9]
    endTime = [1,4,10]
    print(Solution().maxFreeTime(eventTime, k, startTime, endTime))