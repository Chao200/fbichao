---
title: LeetCode Hot 100 Hard(01)
tags:
  - LeetCode
  - Hard
author: fbichao
categories:
  - leetcode
  - Hard
excerpt: LeetCode Hard(01)
math: true
date: 2024-03-01 21:45:00
---
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

## [37. 解数独](https://leetcode.cn/problems/sudoku-solver/description/)

> 编写一个程序，通过填充空格来解决数独问题。

> 数独的解法需 遵循如下规则：

> 数字 1-9 在每一行只能出现一次。
> 数字 1-9 在每一列只能出现一次。
> 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
> 数独部分空格内已填入了数字，空白格用 '.' 表示。

```
遍历行列，对填入的数字判断是否合法即可
```

```C++
class Solution {
public:
    bool isValid(vector<vector<char>>& board, int row, int col, char val)
    {   // 此时还未填入数字

        for (int i = 0; i < 9; ++i)
        {   // 列是否合法
            if (board[i][col] == val) return false;
            // 行是否合法
            if (board[row][i] == val) return false;
        }
      
        // 找到该行列属于第几个小的九宫格
        int startRow = row / 3;
        int startCol = col / 3;

        for (int i = startRow * 3; i < (startRow * 3 + 3); ++i)
        {
            for (int j = startCol * 3; j < (startCol * 3 + 3); ++j)
            {
                if (board[i][j] == val) return false;
            }
        }

        return true;
    }

    bool backtrack(vector<vector<char>>& board)
    {
        for (int i = 0; i < board.size(); ++i)
        {
            for (int j = 0; j < board[0].size(); ++j)
            {
                if (board[i][j] == '.')
                {
                    for (char k = '1'; k <= '9'; ++k)
                    {
                        // 在第 i 行 j 列填入 k，是否合法
                        if (isValid(board, i, j, k))
                        {
                            board[i][j] = k;
                            if (backtrack(board)) return true;
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    void solveSudoku(vector<vector<char>>& board) {
        backtrack(board);
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

## [51. N 皇后](https://leetcode.cn/problems/n-queens/description/)

> 按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

> n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

> 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

> 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

```
![](https://file.fbichao.top/2024/03/94efebaef19d44bfcc3925c47e05bad2.png)
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

```
每个位置放了之后，判断是否合法就行
```

```C++
class Solution {
private:
    vector<vector<string>> res;

public:
    bool isValid(vector<string>& chessBoard, int row, int col, int n)
    {
        // 上方的列
        for (int i = 0; i < row; ++i)
        {
            if (chessBoard[i][col] == 'Q') return false;
        }

        // 左上对角线
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j)
        {
            if (chessBoard[i][j] == 'Q') return false;
        }

        // 右上对角线
        for (int i = row - 1, j = col + 1; i >= 0 && j <= n - 1; --i, ++j)
        {
            if (chessBoard[i][j] == 'Q') return false;
        }

        // 左侧的行不需要考虑，因为一行只放一个
        return true;
    }

    void backtrack(vector<string>& chessBoard, int row, int n)
    {
        if (row == n)
        {
            res.push_back(chessBoard);
            return;
        }
      
        for (int col = 0; col < n; ++col)
        {
            if (isValid(chessBoard, row, col, n))
            {
                chessBoard[row][col] = 'Q';
                backtrack(chessBoard, row + 1, n);
                chessBoard[row][col] = '.';
            }
        }
    }

    vector<vector<string>> solveNQueens(int n) {
        vector<string> chessBoard(n, string(n, '.'));
        backtrack(chessBoard, 0, n);
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

## [115. 不同的子序列](https://leetcode.cn/problems/distinct-subsequences/description/)

> 给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数，结果需要对 109 + 7 取模。

```
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
rabbbit
rabbbit
rabbbit
```

```C++
class Solution {
public:
    int numDistinct(string s, string t) {
        int mod = 1000000007;

        int m = s.size(), n = t.size();

        vector<vector<int>> dp(m + 1, vector<int>(n+1, 0));

        for (int i = 0; i <= m; ++i) dp[i][0] = 1;

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (s[i-1] == t[j-1])
                {
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j];
                }
                else
                {
                    dp[i][j] = dp[i-1][j];
                }
                dp[i][j] %= mod;
            }
        }

        return dp[m][n];
    }
};
```









## [123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/description/)

> 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

> 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
```

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(5, 0));
        // 5 种状态，
        // 第一次不持有、第一次持有、第二次不持有、第二次持有、最后不持有
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[0][2] = 0;
        dp[0][3] = -prices[0];
        dp[0][4] = 0;

        for (int i = 1; i < n; ++i)
        {
            dp[i][0] = dp[i-1][0]; 
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i]);
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] - prices[i]);
            dp[i][4] = max(dp[i-1][4], dp[i-1][3] + prices[i]);
        }

        return dp[n-1][4];
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

## [188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/description/)

> 给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。

> 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。也就是说，你最多可以买 k 次，卖 k 次。

> 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```
输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
```

```C++
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> dp(n, vector<int>(2*k+1, 0));

        // 2k+1 个状态
        dp[0][0] = 0;   // 不持有
        for (int i = 1; i <= 2 * k; ++i)
        {
            // 持有、不持有……
            dp[0][i] = ((i % 2) ? -prices[0] : 0);
        }

        for (int i = 1; i < n; ++i)
        {
            for (int j = 1; j <= 2 * k; ++j)
            {
                dp[i][j] = ((j % 2) ? 
                                max(dp[i-1][j], dp[i-1][j-1] - prices[i]) :
                                max(dp[i-1][j], dp[i-1][j-1]+prices[i]));
            }
        }

        return dp[n-1][2*k];
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

## [297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

> 请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

```
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
```

- 前序遍历递归
- 时间复杂度$O(n)$
- 空间复杂度$O(n)$

```
我们知道需要根据遍历顺序得到二叉树，就必须要有两种遍历方式，但是这里只给了一个 string，所以需要特殊处理
使用 "," 分割每个节点，并将空节点表示为 "nullptr"，这样就得到前序遍历的 string
再将该 string 按照 "," 分隔，可以使用 list<string> 存储
使用递归的前序遍历
```

```C++
class Codec {
public:
    void reserialize(TreeNode* root, string& str)
    {
        if (root == nullptr) str += "nullptr,";
        else
        {
            str += to_string(root->val) + ",";
            reserialize(root->left, str);
            reserialize(root->right, str);
        }
    }

    string serialize(TreeNode* root) {
        string res;
        // 得到加上逗号和 nullptr 的前序遍历 string
        reserialize(root, res);
        return res;
    }

    TreeNode* redeserialize(list<string>& lst)
    {
        if (lst.front() == "nullptr")
        {
            lst.erase(lst.begin());
            return nullptr;
        }

        // 前序遍历的递归形式
        TreeNode* root = new TreeNode(stoi(lst.front()));
        lst.erase(lst.begin());
        root->left = redeserialize(lst);
        root->right = redeserialize(lst);
        return root;
    }
  
    TreeNode* deserialize(string data) {
        list<string> lst;
        string str;
        for (auto c: data)
        {
            if (c == ',')
            {
                lst.push_back(str);
                str.clear();
            }
            else
            {
                str.push_back(c);
            }
        }

        if (!str.empty())
        {
            lst.push_back(str);
            str.clear();
        }
  
        // 去除逗号，得到 lst
        return redeserialize(lst);
    }
};
```

## [301. 删除无效的括号](https://leetcode.cn/problems/remove-invalid-parentheses/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个由若干括号和字母组成的字符串 s ，删除最小数量的无效括号，使得输入的字符串有效。

> 返回所有可能的结果。答案可以按 任意顺序 返回。

```
输入：s = "(a)())()"
输出：["(a())()","(a)()()"]
```

- 回溯
- 时间复杂度$O(n\cdot{2^n})$
- 空间复杂度$O(n^2)$

```
1. 统计左右括号字最少得删除个数（removeInvalidParentheses）
2. 括号合理性判断（isValid）
3. 回溯，对于每个括号，两种 case，删除了不删除；
backtrack(str, index) 表示 str 从 index 开始往下回溯
```

```C++
class Solution {
private:
    vector<string> res;
public:
    vector<string> removeInvalidParentheses(string s) {
        // 得到可删除的左右括号数
        int leftRemoveNum = 0;
        int rightRemoveNum = 0;

        for (char c: s)
        {
            if (c == '(')
            {
                ++leftRemoveNum;
            }
            else if (c == ')')
            {
                if (leftRemoveNum == 0) ++rightRemoveNum;
                else --leftRemoveNum;
            }
        }

        backtrack(s, 0, leftRemoveNum, rightRemoveNum);
        return res;
    }

    bool isValid(const string& str)
    {
        int cnt = 0;

        for (int i = 0; i < str.size(); ++i)
        {
            if (str[i] == '(') ++cnt;
            else if (str[i] == ')')
            {
                --cnt;
                if (cnt < 0) return false;
            }
        }

        return cnt == 0;
    }

    // backtrack(str, index) 对于 str 从 index 开始删除剩余的
    void backtrack(string str, int index, int leftRemoveNum, int rightRemoveNum)
    {
        // 终止条件
        if (leftRemoveNum == 0 && rightRemoveNum == 0)
        {
            if (isValid(str)) res.push_back(str);
            return;
        }

        for (int i = index; i < str.size(); ++i)
        {
            // 去重
            if (i != index && str[i] == str[i-1]) continue;
            // 剪枝
            if ((leftRemoveNum + rightRemoveNum) > str.size() - i) return;
    
            // 删除左括号
            if (leftRemoveNum > 0 && str[i] == '(')
            {                                                 // i 不需要+1，此时 i 就是下一个待删除 index
                backtrack(str.substr(0, i) + str.substr(i+1), i, leftRemoveNum - 1, rightRemoveNum);
            }
            if (rightRemoveNum > 0 && str[i] == ')')
            {
                backtrack(str.substr(0, i) + str.substr(i+1), i, leftRemoveNum, rightRemoveNum - 1);
            }
        }
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

## [332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/description/)

> 给你一份航线列表 tickets ，其中 tickets[i] = [fromi, toi] 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。

> 所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。

> 例如，行程 ["JFK", "LGA"] 与 ["JFK", "LGB"] 相比就更小，排序更靠前。
> 假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。

```
![](https://file.fbichao.top/2024/03/40cdc554cb555d8e9d723af649b832dd.png)
输入：tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
输出：["JFK","ATL","JFK","SFO","ATL","SFO"]
解释：另一种有效的行程是 ["JFK","SFO","ATL","JFK","ATL","SFO"] ，但是它字典排序更大更靠后。
```

```
首先得到每个起点对应终点的航班
根据给定的起点 "JFK" 来遍历是否能够回溯成功
```

```C++
class Solution {
public:
    struct cmp 
    {
        bool operator() (const string& lhs, const string& rhs) const
        {
            return lhs < rhs;
        }
    };

private:
    vector<string> res;
    // 从一个地点，到另一个地点的航班数
    // 并且终点按照字典序排列
    unordered_map<string, map<string, int, cmp>> umap;

public:
    bool backtrack(int num)
    {
        // 两趟航班，三个地点，所以+1
        if (res.size() == num + 1) return true;
      
        // 当前起点对应终点
        for (auto& target: umap[res.back()])
        {
            if (target.second > 0)
            {
                res.push_back(target.first);
                target.second--;
                if (backtrack(num)) return true;
                target.second++;
                res.pop_back();
            }
        }
        return false;
    }

    vector<string> findItinerary(vector<vector<string>>& tickets) {
        for (auto ticket: tickets)
        {
            umap[ticket[0]][ticket[1]]++;
        }

        res.push_back("JFK");
        backtrack(tickets.size());
        return res;
    }
};
```
