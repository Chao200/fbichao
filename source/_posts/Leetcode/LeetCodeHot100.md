---
title: LeetCode Hot 100
tags:
  - LeetCode Hot 100
author: fbichao
categories: leetcode
excerpt: LeetCode Hot 100
math: true
date: 2024-03-01 21:45:00
---

# 简单
## []()










## []()







# 困难
## [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

> 算法的时间复杂度应该为 O(log (m+n)) 。

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

- 排除法二分
- 时间复杂度$O(log(m+n))$ 
- 空间复杂度$O(1)$ 


```C++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int nums1Len = nums1.size();
        int nums2Len = nums2.size();
        // 防止 nums2 索引越界
        if (nums1Len > nums2Len) return findMedianSortedArrays(nums2, nums1);

        int k = (nums1Len + nums2Len + 1) / 2;

        // 排除法的二分，半闭半开区间
        int left = 0, right = nums1Len;
        int m1, m2;
        // 找到数组 nums1 中第一个不小于所有 nums2[m2-1] 的位置
        // 所以 right 不动，left 动
        while (left < right)
        {
            m1 = left + (right - left) / 2;
            m2 = k - m1;
            // 与 m2 - 1 比较
            if (nums1[m1] < nums2[m2-1]) left = m1 + 1;
            else right = m1;
        }

        m1 = left;
        m2 = k-m1;

        // c1 向左找，较大者
        int c1;
        if (m1 <= 0) c1 = nums2[m2 - 1];
        else if (m2 <= 0) c1 = nums1[m1 - 1];
        else c1 = max(nums1[m1-1], nums2[m2-1]);

        if ((nums1Len + nums2Len) % 2) return c1;

        // c2 向右找，较小者
        int c2;
        if (m1 >= nums1Len) c2 = nums2[m2];
        else if (m2 >= nums2Len) c2 = nums1[m1];
        else c2 = min(nums1[m1], nums2[m2]);

        return (c1 + c2) / 2.0;
    }
};
```






## [10. 正则表达式匹配](https://leetcode.cn/problems/regular-expression-matching/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

> '.' 匹配任意单个字符
> '*' 匹配零个或多个前面的那一个元素
> 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

```
输入：s = "aa", p = "a*"
输出：true
解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

- 动规
- 时间复杂度$O(mn)$ 
- 空间复杂度$O(mn)$ 


```C++
class Solution {
public:
    bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        // dp[i][j] 表示 s 的前 i 个和 p 的前 j 个是否匹配
        vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));

        dp[0][0] = true;  // 空字符匹配
        for (int j = 1; j <= n; ++j)
        {
            if (p[j-1] == '*')  // 如果 s 是空，p 要想为空，则必须含有 *，而 * 只能使得紧挨着的字母成为 0 个
            {
                dp[0][j] = dp[0][j-2];  // 所以如果 j-1 是 *，只需要看前 j-2 匹配与否
            }
        }


        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                // if (p[j-1] == '*')
                // {
                //     // 包括 i-1 在内，0 或 1
                //     // 不包括 i-1 在内，2 以上
                //     if (s[i-1] == p[j-2] || p[j-2] == '.') 
                //         dp[i][j] = dp[i][j-2] || dp[i][j-1] || dp[i-1][j];
                //     else dp[i][j] = dp[i][j-2];
                // }
                if (p[j-1] == '*')
                {   // case1. * 作用 0 个
                    if (dp[i][j-2]) dp[i][j] = true;
                    // case2. * 只要前 i-1 个和 前 j 个成立，并且i-1==j-2的值，那就没问题
                    else if (s[i-1] == p[j-2] && dp[i-1][j]) dp[i][j] = true;
                    else if (p[j-2] == '.' && dp[i-1][j]) dp[i][j] = true;
                }
                else    // 如果 j-1 不是 *
                {   // case1. i-1 和 j-1 字母一样
                    if (s[i-1] == p[j-1] && dp[i-1][j-1]) dp[i][j] = true;
                    // case2. j-1 是 .
                    else if (p[j-1] == '.' && dp[i-1][j-1]) dp[i][j] = true;
                }
            }
        }

        return dp[m][n];
    }
};
```






## [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个链表数组，每个链表都已经按升序排列。

> 请你将所有链表合并到一个升序链表中，返回合并后的链表。

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

- 优先队列、归并、依次合并
- 时间复杂度$O(k\cdot{n}\cdot{log(n)})$ 
- 空间复杂度$O(k)$ 


```C++
class Solution {
public:
    
    
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        // 特殊 case
        if (lists.empty()) return nullptr;

        // 虚拟头结点
        ListNode* dummy = new ListNode(0);
        ListNode* cur = dummy;

        // 优先队列，最小堆，链表首元素值小的在顶部
        auto cmp = [](ListNode* a, ListNode* b)
        {
            return a->val > b->val;
        };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq;
        
        // 非空入队
        for (auto node : lists)
        {
            if (node != nullptr)
            {
                pq.push(node);
            }
        }

        // 非空循环
        while (!pq.empty())
        {
            // 取出队首元素
            ListNode* nxt = pq.top();   pq.pop();
            // 存储
            cur->next = nxt;
            cur = cur->next;
            // 对取出队首元素的下一个元素入队
            if (nxt->next!=nullptr)
            {
                pq.push(nxt->next);
            }
        }
        
        return dummy->next;
    }
};
```


## [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

- 栈、动规
- 时间复杂度$O(n)$ 
- 空间复杂度$O(n)$ 


### 栈
```C++
class Solution {
public:
    int longestValidParentheses(string s) {
        int res = 0;
        stack<int> st;
        st.push(-1);

        for (int i = 0; i < s.size(); ++i)
        {
            if (s[i] == '(')
            {
                st.push(i);
            }
            else
            {
                st.pop();
                // 说明栈顶的是非法的
                if (st.empty())
                {
                    st.push(i);
                }
                else
                {
                    res = max(res, i - st.top());
                }
            }
        }

        return res;
    }
};
```

### 动规
```C++
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.size();
        vector<int> dp(n, 0);
        int res = 0;

        for (int i = 1; i < n; ++i)
        {
            // 无法匹配
            if (s[i] == '(') dp[i] = 0;
            else
            {
                if (s[i-1] == '(')
                {
                    // 是否越界
                    if (i >= 2) dp[i] = dp[i-2] + 2;
                    else dp[i] = 2;
                }
                else
                {
                    // i 前一个的匹配项 i-1
                    // 配对的括号的前一个 i-1-dp[i-1] 是 (，和 i 匹配
                    // 还需要看 i-1-dp[i-1] 前面是否还有，如有，则不止一个
                    if ((i - 1- dp[i-1]) >= 0 && s[i-1-dp[i-1]] == '(')
                    {
                        if ((i - 2 - dp[i-1]) >= 0)
                        {
                            dp[i] = dp[i-1] + dp[i-1-dp[i-1]-1] + 2;;
                        }
                        else
                        {
                            dp[i] = dp[i-1] + 2;
                        }
                    }
                }
            }
            res = max(res, dp[i]);
        }

        return res;
    }
};
```




## [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```
![](https://file.fbichao.top/2024/03/cb31bff6e8f6aace0dfc3c998cb92394.png)
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```

- 单调栈（小压大）
- 时间复杂度$O(n)$ 
- 空间复杂度$O(n)$ 


```C++
class Solution {
public:
    int trap(vector<int>& height) {
        // 最少需要三个柱子 小压大
        stack<int> st;
        int n = height.size();
        int res = 0;

        for (int i = 0; i < n; ++i)
        {
            while (!st.empty() && height[i] > height[st.top()])
            {
                int mid_index = st.top(); st.pop();
                int right_index = i;
                if (!st.empty())
                {
                    int left_index = st.top();
                    int h = min(height[left_index], height[right_index]) - height[mid_index];
                    int w = right_index - left_index - 1;
                    res += w * h;
                }
            }
            st.push(i);
        }
        return res;
    }
};
```






## [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
```

- 滑动窗口 + 哈希表
- 时间复杂度$O(len(s))$ 
- 空间复杂度$O(|s|+|t|)$ 


```C++
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> s_map;
        unordered_map<char, int> t_map;

        // 存储 t 的映射
        for (int i = 0; i < t.size(); ++i) t_map[t[i]]++;

        int slow = 0, fast = 0;
        int subLen = s.size() + 1;
        int start = 0;
        int valid = 0;  // 合理性
        while (fast < s.size())
        {
            // 处理快指针，满足加入到映射以及判断合理性
            if (t_map.count(s[fast]))
            {
                s_map[s[fast]]++;
                if (s_map[s[fast]] == t_map[s[fast]])
                {
                    ++valid;
                }
            }


            // 缩小窗口，如果窗口内的值合理了，就要移动 slow
            while (valid == t_map.size())
            {
                // 记录是否有新的结果
                if (subLen > (fast - slow))
                {
                    start = slow;
                    subLen = fast - slow + 1;
                }

                // slow 是否在映射中
                if (s_map.count(s[slow]))
                {
                    // 是否需要减少 valid
                    if (s_map[s[slow]] == t_map[s[slow]]) valid--;
                    s_map[s[slow]]--;
                }
                ++slow;
            }

            fast++;     // 下一个位置
        }

        return subLen == s.size() + 1 ? "" : s.substr(start, subLen);
    }
};
```




## [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

> 求在该柱状图中，能够勾勒出来的矩形的最大面积。

```
![](https://file.fbichao.top/2024/03/c64f84b2f47b85b7fe877581569fc78e.png)
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```

- 单调栈（大压小），不同于 [42-接雨水](#42-接雨水)，这里需要首尾插入 0，因为 while 之后，栈中元素依然可以求结果，而 [42-接雨水](#42-接雨水) 则不需要管栈中剩余元素
- 时间复杂度$O()$ 
- 空间复杂度$O()$ 


```C++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int res = 0;
        stack<int> st;
        // 需要插入 0，使得栈中剩余元素全部出栈
        heights.push_back(0);
        heights.insert(heights.begin(), 0);
        
        for (int i = 0; i < heights.size(); ++i)
        {
            while (!st.empty() && heights[i] < heights[st.top()])
            {
                int right = i;
                int mid = st.top(); st.pop();
                int left = st.top();
                res = max(res, (right-left-1)*heights[mid]);
            }
            st.push(i);
        }

        return res;
    }
};
```






## [85. 最大矩形](https://leetcode.cn/problems/maximal-rectangle/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

```
![](https://file.fbichao.top/2024/03/44aa3873efbb3cc8d9dd9a065726a236.png)
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```

- 压缩数组 + [84-柱状图中最大的矩形](#84-柱状图中最大的矩形)
- 时间复杂度$O(mn)$ 
- 空间复杂度$O(mn)$ 


```C++
class Solution {
public:
    int largestRectangleArea(vector<int> heights) {
        int res = 0;
        stack<int> st;
        // 需要插入 0，使得栈中剩余元素全部出栈
        heights.push_back(0);
        heights.insert(heights.begin(), 0);
        
        for (int i = 0; i < heights.size(); ++i)
        {
            while (!st.empty() && heights[i] < heights[st.top()])
            {
                int right = i;
                int mid = st.top(); st.pop();
                int left = st.top();
                res = max(res, (right-left-1)*heights[mid]);
            }
            st.push(i);
        }

        return res;
    }

    int maximalRectangle(vector<vector<char>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();

        vector<int> nums(n, 0);
        int res = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                nums[j] = matrix[i][j] == '0' ? 0 : nums[j] + 1;
            }
            res  = max(res, largestRectangleArea(nums));
        }
        return res;
    }
};
```






## [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

> 路径和 是路径中各节点值的总和。

> 给你一个二叉树的根节点 root ，返回其 最大路径和 。

```
![](https://file.fbichao.top/2024/03/8563194c270ee51c80efb12d745c2ec2.png)
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```

- 先序遍历的递归
- 时间复杂度$O(n)$ 
- 空间复杂度$O(n)$ 


```C++
class Solution {
private:
    int res = INT_MIN;

public:
    int dfs(TreeNode* root)
    {
        if (root == nullptr) return 0;

        // 不包含 root 的左子树
        int leftSum = max(dfs(root->left), 0);
        // 不包含 root 的右子树
        int rightSum = max(dfs(root->right), 0);

        // 包含 root 的 max
        res = max(res, root->val + leftSum + rightSum);

        // return 只能选择左右子树一条分支
        return root->val + max(leftSum, rightSum);
    }

    int maxPathSum(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```


## [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。


```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

### 优先队列
- 时间复杂度$O(nlogn)$ 
- 空间复杂度$O(n)$ 


```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        auto cmp=[] (const pair<int, int> p1, const pair<int, int>p2)
        {
            return p1.first < p2.first;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> prio_que;

        for (int i = 0; i < k; ++i) prio_que.push({nums[i], i});

        vector<int> ans;
        ans.push_back(prio_que.top().first);
        for (int i = k; i < nums.size(); ++i)
        {
            prio_que.push({nums[i], i});
            while (prio_que.top().second <= i - k)
            {
                prio_que.pop();
            }
            ans.push_back(prio_que.top().first);
        }

        return ans;
    }
};
```


### 单调队列
- 时间复杂度$O(n)$ 
- 空间复杂度$O(k)$ 


```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        
        for (int i = 0; i < k; ++i)
        {
            while (!dq.empty() && nums[dq.back()] <= nums[i])
            {
                dq.pop_back();
            }
            dq.push_back(i);
        }

        vector<int> res;
        res.push_back(nums[dq.front()]);

        for (int i = k; i < nums.size(); ++i)
        {
            while (!dq.empty() && nums[dq.back()] <= nums[i])
            {
                dq.pop_back();
            }
            dq.push_back(i);

            while (dq.front() <= i-k)
            {
                dq.pop_front();
            }

            res.push_back(nums[dq.front()]);
        }

        return res;
    }
};
```



## [312. 戳气球](https://leetcode.cn/problems/burst-balloons/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

> 现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

> 求所能获得硬币的最大数量。

```
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
```

- 动规
- 时间复杂度$O(n^3)$ 
- 空间复杂度$O(n^2)$ 

```
在首尾插入一个气球值为 1，定义 $dp[i][j]$ 表示击破 $i~j$ 之间气球获得的最大值（不包括 $i$ 和 $j$）
假设 $i~j$ 之间最后一个击破的是 $k$，则 $dp[i][j] = dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]$
可以得到状态转移方程
$$
dp[i][j] = max{dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]}
$$
遍历的时候可以先从长度开始，即 3 开始
再确定开始和结束节点，再遍历中间节点
```

```C++
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        nums.insert(nums.begin(), 1);
        nums.push_back(1);

        int n = nums.size();

        vector<vector<int>> dp(n, vector<int>(n, 0));

        // 区间长度
        for (int len = 3; len <= n; ++len)
        {
            // 开始节点
            for (int start = 0; start < n; ++start)
            {
                // 结束节点
                int end = start + len - 1;
                if (end > n - 1) break;
                // 中间节点
                for (int split = start + 1; split < end; ++split)
                {   // 状态转移方程
                    dp[start][end] = max(dp[start][end], dp[start][split] + dp[split][end] + nums[start] * nums[split] * nums[end]);
                }
            }
        }
        return dp[0][n-1];  // 最终的结果是不包括首尾的
    }
};
```






## []()

>

```

```

- 
- 时间复杂度$O()$ 
- 空间复杂度$O()$ 


```C++

```




## []()

>

```

```

- 
- 时间复杂度$O()$ 
- 空间复杂度$O()$ 


```C++

```






## []()

>

```

```

- 
- 时间复杂度$O()$ 
- 空间复杂度$O()$ 


```C++

```






## []()

>

```

```

- 
- 时间复杂度$O()$ 
- 空间复杂度$O()$ 


```C++

```