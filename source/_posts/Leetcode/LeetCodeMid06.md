---
title: LeetCode Mid(6)
tags:
  - LeetCode
  - Mid
author: fbichao
categories:
  - leetcode
  - Mid
excerpt: LeetCode Mid(6)
math: true
date: 2024-03-25 21:45:00
---
## [322. 零钱兑换](https://leetcode.cn/problems/coin-change/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

> 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

> 你可以认为每种硬币的数量是无限的。

```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

- 动态规划
- 时间复杂度 $O(mn)$，n 是硬币个数
- 空间复杂度 $O(m)$，m 是金额

```
动态规划，求 min，初始化设置成较大值
遍历两个东西
```

```C++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        // 记录组成 amount 大小的个数
        vector<int> dp(amount+1, amount+1);
      
        dp[0] = 0;

        // 遍历大小
        for (int i = 1; i <= amount; ++i)
        {
            // 遍历硬币面值
            for (int j = 0; j < coins.size(); ++j)
            {
                // 如果可以构成
                if (coins[j] <= i)
                {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
};
```

## [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

> 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

> 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。

```
![](https://file.fbichao.top/2024/03/34d3dd51f0efb8abb9d87e1d0755709a.png)
输入: root = [3,2,3,null,3,null,1]
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7
```

- 动态规划
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
动规的数组作为返回值，先序遍历的递归形式
```

```C++
class Solution {
public:
    vector<int> f(TreeNode* root)
    {
        // [偷、不偷]
        if (root == nullptr) return {0, 0};
      
        // 左右子树结果
        vector<int> left = f(root->left);
        vector<int> right = f(root->right);

        // ① 偷当前 root，那么左右两边子节点偷不了
        // ② 不偷当前 root，那么左右两边子节点可以偷，也可以不偷，取左右子树偷和不偷的最大值
        return {root->val + left[1] + right[1],
                    max(left[0], left[1]) + max(right[0], right[1])};
    }

    int rob(TreeNode* root) {
        auto res = f(root);
        return max(res[0], res[1]);
    }
};
```

## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

- 优先队列
- 时间复杂度 $O(n\cdot{logk})$，n 是数组长度，堆的大小是 k，所以每次插入是 logk 复杂度
- 空间复杂度 $O(n)$，哈希表是 $O(n)$，堆是 $O(k)$

```
优先队列存储元素及其频率，大根堆
```

```C++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 统计频率
        unordered_map<int, int> umap;
        for (auto n: nums) umap[n]++;

        // 大根堆
        auto cmp = [](pair<int,int>&m, pair<int,int>&n) {return m.second > n.second;};
        priority_queue<pair<int, int>, vector<pair<int,int>>, decltype(cmp)> q;

        // 全部入优先队列，保持个数为 k 个
        for (auto [num, count]: umap)
        {
            q.push({num, count});
            if (q.size() > k) q.pop();
        }

        // 存储结果
        vector<int> res;
        while (!q.empty())
        {
            res.push_back(q.top().first);
            q.pop();
        }

        return res;
    }
};
```

## [394. 字符串解码](https://leetcode.cn/problems/decode-string/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个经过编码的字符串，返回它解码后的字符串。

> 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

> 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

> 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

```
输入：s = "3[a2[c]]"
输出："accaccacc"
```

- 栈模拟
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
使用栈进行模拟，存储每次的字符串片段和数字
```

```C++
class Solution {
public:
    // s 字符串重复 n 次
    string repeat(string s, int n)
    {
        string ans = "";
        while (n--) ans += s;
        return ans;
    }

    string decodeString(string s) {
        // 存储数字之前的字符串
        stack<string> str_st;
        // 存储当前数字
        stack<int> num_st;
        string res = "";
        int cur_num = 0;

        for (char c: s)
        {
            if (c >= '0' && c <= '9')
            {
                cur_num = cur_num * 10 + c - '0';
            }
            else if (c == '[')
            {
                // 数字之前的字符串
                str_st.push(res);
                // 当前数字
                num_st.push(cur_num);
                res = "";
                cur_num = 0;
            }
            else if (c == ']')
            {
                string str = str_st.top(); str_st.pop();
                int num = num_st.top(); num_st.pop();
                res = str + repeat(res, num);
            }
            else 
            {
                res += c;
            }
        }

        return res;
    }
};
```

## [399. 除法求值](https://leetcode.cn/problems/evaluate-division/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

> 另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

> 返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

> 注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

> 注意：未在等式列表中出现的变量是未定义的，因此无法确定它们的答案。

```

```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++

```

## [406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 假设有打乱顺序的一群人站成一个队列，数组 people 表示队列中一些人的属性（不一定按顺序）。每个 people[i] = [hi, ki] 表示第 i 个人的身高为 hi ，前面 正好 有 ki 个身高大于或等于 hi 的人。

> 请你重新构造并返回输入数组 people 所表示的队列。返回的队列应该格式化为数组 queue ，其中 queue[j] = [hj, kj] 是队列中第 j 个人的属性（queue[0] 是排在队列前面的人）。

```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
```

- 排序
- 时间复杂度 $O(n^2)$，排序是 $nlogn$，遍历加插入 $O(n^2)
- 空间复杂度 $O(n)$，排序需要 $logn$，存储结果需要 $O(n)$

```
两个变化，固定一个顺序，再按照另一个排
在这里，身高降序，再按第二个元素插入
```

```C++
class Solution {
public:
    static bool cmp(const vector<int>& a, const vector<int>&b)
    {
        // 按照第一个元素降序，相等时，按照第二个元素升序
        if (a[0] == b[0]) return a[1] < b[1];
        return a[0] > b[0];
    }

    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(), people.end(), cmp);
        vector<vector<int>> res;
        for (vector<int>& p: people)
        {
            // 按照第二个元素大值，插入，因为后面的元素不可能比前面的大，所以可以
            res.insert(res.begin() + p[1], p);
        }
        return res;
    }
};
```

## [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

```
输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。
```

- 动态规划、0-1 背包问题
- 时间复杂度 $O(n\cdot{target})$，n 是数组长度
- 空间复杂度 $O(target)$

```
dp[i] 表示容量为 i 的背包最多可以装多少东西
从前向后变量物品，从后向前遍历容量
```

```C++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        for (auto n: nums) sum += n;
        if (sum % 2) return false;

        int target = sum / 2;
        vector<int> dp(target + 1, 0);

        // 遍历物品
        for (int i = 0; i < nums.size(); ++i)
        {
            // 从后向前遍历容量
            for (int j = target; ; --j)
            {
                // 容量大于等于重量
                if (j >= nums[i])
                    dp[j] = max(dp[j], dp[j-nums[i]]+ nums[i]);
                else
                    break;
            }
        }

        return dp[target] == target;
    }
};
```

## [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

> 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

```
![](https://file.fbichao.top/2024/03/0178bb47e86238fb1320c197f31e44f2.png)
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
```

### DFS

- 时间复杂度 $O(n^2)$，n 是节点个数，每个节点都要求路径和
- 空间复杂度 $O(n)$

```
先序遍历的递归形式，
```

```C++
class Solution {
public:
    int rootSum(TreeNode* root, long targetSum)
    {
        if (root == nullptr) return 0;

        int count = 0;
        if (root->val == targetSum) count++;

        count += rootSum(root->left, targetSum - root->val);
        count += rootSum(root->right, targetSum - root->val);
        return count;
    }

    int pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return 0;

        int count = rootSum(root, targetSum);
        count += pathSum(root->left, targetSum);
        count += pathSum(root->right, targetSum);
        return count;
    }
};
```

### 前缀和

- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
求出从根节点到每个节点的和，用两个节点相减，就得到差值
```

```C++
class Solution {
private:
    unordered_map<long long, int> prefix;

public:
    int dfs(TreeNode* root, long long cur, int targetSum)
    {
        if (root == nullptr) return 0;

        int count = 0;
        // 当前的和
        cur += root->val;

        // prefix[0] = 1
        if (prefix.count(cur - targetSum))
        {
            count = prefix[cur - targetSum];
        }

        // 前缀和为 cur 的路径个数
        prefix[cur]++;
        // 左子树
        count += dfs(root->left, cur, targetSum);
        // 右子树
        count += dfs(root->right, cur, targetSum);
        // 前缀和的节点之间需要具有父子关系，遍历完后，需要移除
        prefix[cur]--;

        return count;
    }

    int pathSum(TreeNode* root, int targetSum) {
        // cur - targetSum == 0 时
        prefix[0] = 1;
        return dfs(root, 0, targetSum);
    }
};
```

## [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

> 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

```
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
```

- 双指针、定长滑动窗口
- 时间复杂度 $O(n+m)$，n 是 s 长度，m 是 p 长度
- 空间复杂度 $O(\Sigma)$，$\Sigma$ 是字符个数

```
正常扩张，每次扩张的字母需要判断是否在 base 中，合法性判断
收缩，窗口大小超过 p.size()
```

```C++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        // 基准
        unordered_map<char, int> base, window;
        for (char ch: p) base[ch]++;

        // 双指针
        int left = 0, right = 0;
        // 存储结果
        vector<int> res;
        // 合法个数
        int valid = 0;

        while (right < s.size())
        {
            // 扩张
            char r_char = s[right];
            right++;

            // 如果字符在 base 中
            if (base.count(r_char))
            {
                window[r_char]++;
                // 是否 valid，即个数一致
                if (window[r_char] == base[r_char]) ++valid;
            }

            // 当个数达到 p 的大小，窗口收缩条件
            if (right - left == p.size())
            {
                // 合法个数和 base 个数是否相等
                if (valid == base.size()) res.push_back(left);

                char l_char = s[left];
                left++;
                if (base.count(l_char))
                {
                    // 如果是合法，需要--
                    if (window[l_char] == base[l_char]) valid--;
                    // 减少计数
                    window[l_char]--;
                }
            }

        }

        return res;
    }
};
```

## [494. 目标和](https://leetcode.cn/problems/target-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个非负整数数组 nums 和一个整数 target 。

> 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

> 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
> 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

```
输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

- 动态规划、0-1 背包问题
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
转化为背包问题
```

```C++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        for (int n: nums) sum += n;

        // 假设正数和为 x，负数和为 y，则 x + y = target
        // x - y = sum
        // x = (sum + target) / 2
        if (sum + target < 0) return 0;
        if ((sum + target) % 2) return 0;
      
        int bag = (sum + target) / 2;
        vector<int> dp(bag + 1, 0);
        dp[0] = 1;  // 目标和为 0 是的 case
        // 遍历物品
        for (int i = 0; i < nums.size(); ++i)
        {
            // 遍历背包
            for (int j = bag; j >= nums[i]; --j)
            {
                // 如果 j 等于 nums[i] 应是一种 case
                dp[j] += dp[j-nums[i]];
            }
        }

        return dp[bag];
    }
};
```

## [538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

```
![](https://file.fbichao.top/2024/03/0e44e54ce279a77731e6bbff0b54d2c4.png)
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
```

- 反中序递归遍历
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
反中序递归遍历
```

```C++
class Solution {
private:
    int sum = 0;

public:
    // 中序遍历得到的升序，反中序得到的是降序
    // 右中左遍历
    void dfs(TreeNode* root)
    {
        if (root == nullptr) return ;
        dfs(root->right);   // 右
        sum += root->val;   // 中间
        root->val = sum;    // 更新节点值
        dfs(root->left);    // 左
    }

    TreeNode* convertBST(TreeNode* root) {
        dfs(root);
        return root;
    }
};
```

## [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

> 子数组是数组中元素的连续非空序列。

```
输入：nums = [1,2,3], k = 3
输出：2
```

- 前缀和
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
使用前缀和，由于只需要统计次数，使用哈希表，空间换时间
```

```C++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> umap;   // 存储前缀和
        umap[0] = 1;    // preSum - k 为 0 时

        int preSum = 0;
        int ans = 0;

        for (int i = 0; i < nums.size(); ++i)
        {
            preSum += nums[i];
            if (umap.count(preSum - k))
            {
                ans += umap[preSum - k];
            }
            umap[preSum]++;
        }

        return ans;
    }
};
```

## [581. 最短无序连续子数组](https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。

> 请你找出符合题意的 最短 子数组，并输出它的长度。

```
输入：nums = [2,6,4,8,10,9,15]
输出：5
解释：你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
```

- 一次遍历
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
可以排序，创建一个新的数组排序，比对新老数组首尾元素是否相等，也可以找到左右边界
一次遍历，利用 
- nums[0] < nums[1] < …… < nums[left-1] < minVal < nums[left]
    - 左边已经有序，从右向左遍历即可
- nums[n-1] > nums[n-2] > …… > nums[right+1] > maxVal > nums[right]
    - 右边已经有序，从左向右遍历即可
```

```C++
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int n = nums.size();
      
        // 区间是 【left， right】
        int left = 0, right = -1;   // 如果是升序的，则 right-left+1 是 0
        // 假设中段区间的最大值是 maxval，最小值是 minval
        int maxval = nums[0];
        int minval = nums[n-1];

        for (int i = 0; i < n; ++i)
        {
            // nums[n-1] > nums[n-2] > …… > nums[R+1] > maxVal > nums[R]
            // 从左向右遍历，最后一个小于 maxVal 的为右边界
            if (nums[i] < maxval)
            {   // 更新右边界
                right = i;
            }
            else
            {   // 更新最大值
                maxval = nums[i];
            }

            // nums[0] < nums[1] < …… < nums[left - 1] < minVal < nums[left]
            // 从右向左遍历，最后一个大于 minVal 的为左边界
            if (nums[n-i-1] > minval)
            {   // 更新左边界
                left = n - i - 1;
            }
            else
            {   // 更新最小值
                minval = nums[n-i-1];
            }
        }

        return right - left + 1;
    }
};
```

## [621. 任务调度器](https://leetcode.cn/problems/task-scheduler/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表，用字母 A 到 Z 表示，以及一个冷却时间 n。每个周期或时间间隔允许完成一项任务。任务可以按任何顺序完成，但有一个限制：两个 相同种类 的任务之间必须有长度为 n 的冷却时间。

> 返回完成所有任务所需要的 最短时间间隔 。

```
输入：tasks = ["A","A","A","B","B","B"], n = 2
输出：8
解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B
     在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。 
```

- 构造
- 时间复杂度 $O(|tasks|, \Sigma)$，|tasks| 是任务总数，$Sigma$ 是字符集
- 空间复杂度 $O(\Sigma)$

```
maxExec：最长的任务的长度
maxCount：最长的任务的个数
参考[题解](https://leetcode.cn/problems/task-scheduler/solutions/509687/ren-wu-diao-du-qi-by-leetcode-solution-ur9w/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)
```

```C++
class Solution {
public:
    int leastInterval(vector<char>& tasks, int n) {
        unordered_map<char, int> umap;
        for (char c: tasks) umap[c]++;

        // (maxExec - 1) * (n + 1) + maxCount;
        int maxExec = 0;
        for (auto p: umap)
        {
            if (maxExec < p.second)
            {
                maxExec = p.second;
            }
        }

        int maxCount = 0;
        for (auto p: umap)
        {
            if (p.second == maxExec)
            {
                maxCount++;
            }
        }

        return max((maxExec - 1) * (n + 1) + maxCount, static_cast<int>(tasks.size()));
    }
};
```

## [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。

> 回文字符串 是正着读和倒过来读一样的字符串。

> 子字符串 是字符串中的由连续字符组成的一个序列。

> 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

```
输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

- 动态规划
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n^2)$

```
- dp[i][j] 表示 s[i]:s[j] 是否为回文串
- 状态转移 dp[i][j] = dp[i+1][j-1]
- 遍历方向是 i 从大到小，j 是从小到大，由状态方程得到
```

```C++
class Solution {
public:
    int countSubstrings(string s) {
        int n = s.size();
        // dp[i][j] 表示 s[i]:s[j] 是否为回文串
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        int ans = 0;

        // 根据递推公式，i+1 和 j-1
        // 所以 i 需要从后往前，即从大到小遍历
        // j 从小到大遍历
        for (int i = n - 1; i >= 0; --i)
        {
            for (int j = i; j < n; ++j)
            {
                if (s[i] == s[j])
                {
                    if (j - i <= 2) dp[i][j] = true;
                    else
                    {   // 递推公式
                        dp[i][j] = dp[i+1][j-1];
                    }
                }
                if (dp[i][j]) ans++;
            }
        }
        return ans;
    }
};
```

## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

```
输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
```

- 单调栈
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
找大元素，就小压大
```

```C++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res(n, 0);
        stack<int> st;
        for (int i = 0; i < n; ++i)
        {
            // 如果入栈元素大于栈顶元素，出栈
            while (!st.empty() && temperatures[i] > temperatures[st.top()])
            {
                // 入栈元素和出栈元素构成结果
                res[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);
        }

        return res;
    }
};
```
