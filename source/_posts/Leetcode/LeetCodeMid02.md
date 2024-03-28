---
title: LeetCode Mid(2)
tags:
  - LeetCode
  - Mid
author: fbichao
categories:
  - leetcode
  - Mid
excerpt: LeetCode Mid(2)
math: true
date: 2024-03-08 21:45:00
---
## [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

> 如果数组中不存在目标值 target，返回 [-1, -1]。

> 你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

- 二分法
- 时间复杂度 $O(logn)$
- 空间复杂度 $O(1)$

```
二分法找左右边界
```

```C++
class Solution {
public:
    int binarySearchLeft(vector<int>& nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        while (left < right)
        {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target)
            {
                right = mid;
            }
            else if (nums[mid] < target)
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }
        if (nums[left] == target) return left;
        return -1;
    }

    int binarySearchRight(vector<int>& nums, int target)
    {
        int left = 0, right = nums.size() - 1;
        while (left < right)
        {
            int mid = left + (right - left + 1) / 2;
            if (nums[mid] > target)
            {
                right = mid - 1;
            }
            else if (nums[mid] < target)
            {
                left = mid;
            }
            else
            {
                left = mid;
            }
        }
        if (nums[left] == target) return left;
        return -1;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.size() == 0) return {-1, -1};
        int left = binarySearchLeft(nums, target);
        int right = binarySearchRight(nums, target);
        return {left, right};
    }
};
```

## [39. 组合总和](https://leetcode.cn/problems/combination-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

> candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。

> 对于给定的输入，保证和为 target 的不同组合数少于 150 个。

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
解释：
2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
7 也是一个候选， 7 = 7 。
仅有这两种组合。
```

- 回溯
- 时间复杂度 $O(n\cdot{2^n})$
- 空间复杂度 $O(target)$

```
回溯下一个数字，遍历当前数字及其后面的
```

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& candidates, int target, int sum, int index)
    {
        if (sum > target) return;
        if (sum == target)
        {
            res.push_back(path);
            return;
        }

        for (int i = index; i < candidates.size(); ++i)
        {
            sum += candidates[i];
            path.push_back(candidates[i]);
            backtrack(candidates, target, sum, i);
            path.pop_back();
            sum -= candidates[i];
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        backtrack(candidates, target, 0, 0);
        return res;
    }
};
```

## [46. 全排列](https://leetcode.cn/problems/permutations/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

- 回溯
- 时间复杂度 $O(n\cdot{n!})$
- 空间复杂度 $O(n)$

```
从头回溯每个数字，所以需要 used 数组，遍历每个数字
```

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& nums, vector<bool>& used)
    {
        if (path.size() == nums.size())
        {
            res.push_back(path);
            return;
        }

        for (int i = 0; i < nums.size(); ++i)
        {
            if (used[i] == false)
            {
                used[i] = true;
                path.push_back(nums[i]);
                backtrack(nums, used);
                path.pop_back();
                used[i] = false;
            }
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        backtrack(nums, used);
        return res;
    }
};
```

## [48. 旋转图像](https://leetcode.cn/problems/rotate-image/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

> 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

```
![](https://file.fbichao.top/2024/03/c6963849da5fdd7a0f28ffdcc8adc88e.png)
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

- 两次反转数组
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(1)$

```
垂直+斜对角线反转
```

```C++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        // 斜对角线反转
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        // 垂直反转
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < (cols / 2); ++j)
            {
                swap(matrix[i][j], matrix[i][cols-j-1]);
            }
        }
    }
};
```

## [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

> 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

- 排序+哈希表
- 时间复杂度 $O(nklogk)$，n 是字符串个数，k 是字符串平均长度
- 空间复杂度 $O(nk)$

```
排序后的字符串作为 key，异位词作为 value
```

```C++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // 哈希表
        unordered_map<string, vector<string>> umap;

        for (auto s: strs)
        {
            string temp = s;
            sort(s.begin(), s.end());
            umap[s].push_back(temp);
        }

        vector<vector<string>> res;
        for (auto mp: umap)
        {
            res.push_back(mp.second);
        }
        return res;
    }
};
```

## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

> 子数组是数组中的一个连续部分。

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```
dp 数组定义为以 nums[i] 的连续子数组的最大和是多少
```

```C++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        // 以 nums[i] 结尾的最大连续子数组和
        vector<int> dp(n, 0);

        dp[0] = nums[0];

        int res = dp[0];

        for (int i = 1; i < n; ++i)
        {
            dp[i] = max(dp[i-1]+nums[i], nums[i]);

            res = res < dp[i] ? dp[i] : res;
        }

        return res;
    }
};
```

## [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

> 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。

```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

- 贪心
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
不断更新可到达的最远距离，看是否大于当前所在位置
相当于一个在前面修路，一个在后面走，想要走路必须要有路才行
参考[题解](https://leetcode.cn/problems/jump-game/solutions/24322/55-by-ikaruga/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)
```

```C++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 当前可到达的最远距离
        int maxVal = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            // 如果当前位置大于了可达到的最远距离，false
            if (i > maxVal) return false;
            // 如果最远距离已经可以到达最后一个元素，true
            if (maxVal >= nums.size() - 1) return true;
            // 更新 maxVal
            maxVal = max(maxVal, i + nums[i]);
        }
        return true;
    }
};
```

## [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

- 排序
- 时间复杂度 $O(nlogn)$
- 空间复杂度 $O(logn)$

```
首先固定一个坐标，然后再比较另一个坐标就方便了
```

```C++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) return {};
      
        // 排序
        auto cmp = [](vector<int>& t1, vector<int>&t2)
        {
            return t1[0] < t2[0];
        };
        sort(intervals.begin(), intervals.end(), cmp);
        vector<vector<int>> res;

        for (int i = 0; i < intervals.size(); ++i)
        {
            // 当前区间左右坐标
            int cur_L = intervals[i][0];
            int cur_R = intervals[i][1];
            // 如果 res 为空，或者不重叠
            if (!res.size() || res.back()[1] < cur_L)
            {
                res.push_back({cur_L, cur_R});
            }
            else
            {
                res.back()[1] = max(res.back()[1], cur_R);
            }
        }
        return res;
    }
};
```

## [62. 不同路径](https://leetcode.cn/problems/unique-paths/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

> 问总共有多少条不同的路径？

```
![](https://file.fbichao.top/2024/03/b9a8438247c3121b79b9e59d12a622a3.png)
输入：m = 3, n = 7
输出：28
```

- 动规
- 时间复杂度 $O(mn)$
- 空间复杂度 $O(mn)$

```
动规，当前节点只能通过上面和左面过来，边界另外考虑
```

```C++
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n, 0));

        // 初始化
        dp[0][0] = 1;
        for (int i = 0; i < m; ++i) dp[i][0] = 1;
        for (int i = 0; i < n; ++i) dp[0][i] = 1;

        // 遍历
        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }

        return dp[m-1][n-1];
    }
};
```

## [64. 最小路径和](https://leetcode.cn/problems/minimum-path-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

> 说明：每次只能向下或者向右移动一步。

```
![](https://file.fbichao.top/2024/03/b0ba5a62329131c43071badf5b538972.png)
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

- 动规
- 时间复杂度 $O(mn)$
- 空间复杂度 $O(mn)$

```
与 [62. 不同路径](#62-不同路径) 类似，不同之处在于去最小 min
```

```C++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();

        vector<vector<int>> dp(rows, vector<int>(cols, 0));
      
        // 初始化
        dp[0][0] = grid[0][0];
        for (int i = 1; i < rows; ++i) dp[i][0] = dp[i-1][0] + grid[i][0];
        for (int i = 1; i < cols; ++i) dp[0][i] = dp[0][i-1] + grid[0][i];

        // 状态转移
        for (int i = 1; i < rows; ++i)
        {
            for (int j = 1; j < cols; ++j)
            {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + + grid[i][j];
            }
        }

        return dp[rows-1][cols-1];
    }
};
```
