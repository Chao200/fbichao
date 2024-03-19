---
title: LeetCode Hot 100 Mid(1)
tags:
  - LeetCode
  - Hot100
  - Mid
author: fbichao
categories:
  - leetcode
  - Hot100
  - Mid
excerpt: LeetCode Hot 100 Mid(1)
math: true
date: 2024-03-05 21:45:00
---
## [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

> 请你将两个数相加，并以相同形式返回一个表示和的链表。

> 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
![](https://file.fbichao.top/2024/03/565a891f15dc13e222df608d8df1beed.png)
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```

- 模拟题
- 时间复杂度 $O(max(len(l1), len(l2)))$
- 空间复杂度 $O(1)$

```
模拟计算两个数的和
每次需要取两个数，注意如果为空，则是 0
非空节点在循环结束需要移动
```

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 新的虚拟头结点
        ListNode* sumHead = new ListNode(-1);
        // 用于存储节点
        ListNode* cur = sumHead;

        // 基数和进位
        int base = 0, carry = 0;
        while (carry || l1 || l2)
        {
            // 两个待计算的数，若为空节点，则返回 0
            int num1 = l1 == nullptr ? 0 : l1->val;
            int num2 = l2 == nullptr ? 0 : l2->val;
            // 新的 基数和进位
            base = (num1 + num2 + carry) % 10;
            carry = (num1 + num2 + carry) / 10;

            // 添加新节点
            cur->next = new ListNode(base);
            cur = cur->next;

            // 非空移动
            if (l1) l1 = l1->next;
            if (l2) l2 = l2->next;
        }

        return sumHead->next;
    }
};
```

## [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个字符串 s ，请你找出其中不含有重复字符的最长子串的长度。

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

- 滑动窗口、快慢双指针、unordered_map
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$，可以认为是常数的

```
使用哈希表和滑动窗口
快指针每次循环都会懂，慢指针只有在收缩窗口时才会动，并且一般是 while
每次计算新的 res
```

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if (s.size() == 0) return 0;

        unordered_map<char, int> umap;

        int res = 0;
      
        int slow = -1;
        for (int fast = 0; fast < s.size(); ++fast)
        {
            umap[s[fast]]++;
            // while 循环，不是 if
            while (umap[s[fast]] > 1)
            {
                ++slow;
                umap[s[slow]]--;
            }
            // 计算新的res
            res = max(res, fast - slow);
        }

        return res;
    }
};
```

## [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串 s，找到 s 中最长的回文子串。

> 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

- 动规
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n^2)$

```
dp[i][j] 表示 i~j 是否为回文串
外层循环是 j，确定 j 后
i 追赶至 j
状态转移方程
$$
dp[i][j]=
\begin{cases}
dp[i+1][j-1]& \text{s[i]==s[j]}\\
true& \text{j-i<=2}
\end{cases}
$$
```

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        // base case
        int n = s.size();
        if (n == 0) return "";

        // dp[i][j] 从 i~j 是否为回文串
        vector<vector<bool>> dp(n, vector<bool>(n, false));
        dp[0][0] = true;
        // 因为结果是 string，所以需要记录开始的地方和长度
        int start = 0, len = 1;

        // 双指针，一个在后面，一个在前面追
        for (int j = 1; j < n; ++j)
        {
            for (int i = 0; i < j; ++i)
            {
                // 如果相等
                if (s[i] == s[j])
                {
                    // case1. 长度小于等于 3
                    if (j - i <= 2) dp[i][j] = true;
                    else  // case2. 长度大于 3
                    {
                        dp[i][j] = dp[i+1][j-1];
                    }
                }

                // 更新长度
                if (dp[i][j] && (j - i + 1) > len)
                {
                    start = i;
                    len = j - i + 1;
                }
            }
        }

        return s.substr(start, len);
    }
};
```

## [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

> 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

> 返回容器可以储存的最大水量。

```
![](https://file.fbichao.top/2024/03/865ea140915862e281d93778d4b4820e.png)
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```

- 相向双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
移动最短边的指针，意味着不可能从这个边得到更大的值，理由见[leetcode题解](https://leetcode.cn/problems/container-with-most-water/solutions/207215/sheng-zui-duo-shui-de-rong-qi-by-leetcode-solution/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)
```

```C++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = height.size() - 1;
        int res = 0;

        // 相向双指针
        while (left < right)
        {
            // 取最小的高
            int h = min(height[left], height[right]);
            // 面积
            int temp = h * (right - left);
            res = max(res, temp);
            // 移动最短边
            if (h == height[left])
            {
                ++left;
            }
            else
            {
                --right;
            }
        }
        return res;
    }
};
```

## [15. 三数之和](https://leetcode.cn/problems/3sum/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

> 你返回所有和为 0 且不重复的三元组。

> 注意：答案中不可以包含重复的三元组。

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
```

- 三指针、相向双指针、排序
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(logn)$，如果不计答案数组，并且可以在 nums 上原地排序，是 $logn$，不可以修改 nums 的话，就是 $O(n)$

```
我们需要三个数才能计算得到结果，所示 $O(n^3)$ 的遍历，但是可以使用双指针优化为 $O(n^2)$
不过需要**先排序**，排序的目的一方面是为了便于去重，另一方面可以更快的停止双指针的移动
固定一个指针，其余两个是相向双指针
```

```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        // 排序
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<vector<int>> res;

        // 第一个指针
        for (int i = 0; i < n; ++i)
        {
            // 去重
            if (i > 0 && nums[i] == nums[i-1]) continue;

            // 后两个指针
            int left = i + 1;
            int right = n - 1;
            while (left < right)
            {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) --right;
                else if (sum < 0) ++left;
                else
                {
                    res.push_back({nums[i], nums[left], nums[right]});

                    // 去重
                    while (left < right && nums[left] == nums[left + 1]) ++left;
                    while (left < right && nums[right] == nums[right - 1]) --right;
                    ++left;
                    --right;
                }
            }
        }

        return res;
    }
};
```

## [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

> 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![](https://file.fbichao.top/2024/03/fa6bfcfecc9e4e2490bd1ae1e5a44d2b.png)

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

- 回溯
- 时间复杂度 $O(3^m\cdot{4^n})$，m 是三个字母的按键数，n 是 4 个字母按键数
- 空间复杂度 $O(m+n)$

```
构建按键和字母映射的数据结构
回溯按键，遍历字母
```

```C++
class Solution {
private:
    vector<string> res;
    string path;

    // 电话按键
    const string letterMap[10] = {
        "", // 0
        "", // 1
        "abc", // 2
        "def", // 3
        "ghi", // 4
        "jkl", // 5
        "mno", // 6
        "pqrs", // 7
        "tuv", // 8
        "wxyz", // 9
    };

public:
    void backtrack(string digits, int index)
    {
        // 终止条件
        if (index == digits.size())
        {
            res.push_back(path);
            return;
        }

        int number = digits[index] - '0';
        string letter = letterMap[number];
        // 对一个按键的所有字母遍历
        for (int i = 0; i < letter.size(); ++i)
        {
            path.push_back(letter[i]);
            backtrack(digits, index+1); // 下一个按键
            path.pop_back();
        }
    }

    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) return res;
        backtrack(digits, 0);
        return res;
    }
};
```

## [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

```
![](https://file.fbichao.top/2024/03/5d51a098c56eb5ad85ed2d1128e70adf.png)
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

- 双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
快慢双指针实现一趟扫描
```

```C++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummpy = new ListNode(-1);
        dummpy->next = head;
        ListNode* fast = dummpy;
        ListNode* slow = dummpy;

        // 快指针先动
        while (n--)
        {
            fast = fast->next;
        }

        while (fast->next)
        {
            slow = slow->next;
            fast = fast->next;
        }

        ListNode* del = slow->next;
        slow->next = slow->next->next;
        delete del;

        return dummpy->next;
    }
};
```

## [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```
回溯下一个括号，遍历左右括号，先左后右
```

```C++
class Solution {
private:
    vector<string> res;
    string path;

public:
    void backtrack(int n, int left, int right)
    {
        if (path.size() == 2 * n)
        {
            res.push_back(path);
            return;
        }

        if (left < right) return;

        // 先加左括号
        if (left < n)
        {
            path.push_back('(');
            backtrack(n, left + 1, right);
            path.pop_back();
        }

        // 后加右括号
        if (right < left)
        {
            path.push_back(')');
            backtrack(n, left, right + 1);
            path.pop_back();
        }
    }

    vector<string> generateParenthesis(int n) {
        backtrack(n, 0, 0);
        return res;
    }
};
```

## [31. 下一个排列](https://leetcode.cn/problems/next-permutation/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

> 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
> 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

> 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
> 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
> 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
> 给你一个整数数组 nums ，找出 nums 的下一个排列。

> 必须 原地 修改，只允许使用额外常数空间。

```
输入：nums = [1,2,3,8,5,7,6,4]
输出：[1,2,3,8,6,4,5,7]
```

- 两次遍历
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
从后到前，两次扫描
参考[题解](https://leetcode.cn/problems/next-permutation/solutions/80560/xia-yi-ge-pai-lie-suan-fa-xiang-jie-si-lu-tui-dao-/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

1. 从后至前，找到第一对递增的 (i,j)，即 i+1==j，且nums[i] < nums[j]
2. 从后至 j 找到第一个大于 i 的数 k，即 nums[k] > nums[i]，且 k>=j
3. 此时 j 到末尾是降序，反转为升序
```

```C++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return;

        int i = n - 2, j = n - 1, k = n - 1;
      
        // 1. 查找第一对递增序列
        while (i >= 0 && nums[i] >= nums[j])
        {
            --i;
            --j;
        }

        // 2. 找 k
        if (i >= 0) // 不是最大的排列
        {
            while (nums[i] >= nums[k])
            {
                --k;
            }
            swap(nums[i], nums[k]);
        }

        // 3. 反转
        reverse(nums.begin() + j, nums.end());
    }
};
```

## [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 整数数组 nums 按升序排列，数组中的值 互不相同 。

> 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

> 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

> 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

- 二分查找
- 时间复杂度 $O(logn)$
- 空间复杂度 $O(1)$

```
对一个有序数组旋转后进行二分查找，从一个有序数组变成两个有序
需要判断 mid 在左半区还是右半区即可
```

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            if (nums[left] <= nums[mid]) {
                // 在左半区
                if (target >= nums[left] && target <= nums[mid])
                {
                    right = mid - 1;
                }
                else    // 右半区
                {
                    left = mid + 1;
                }
            }
            else {
                // 右半区
                if (target >= nums[mid] && target <= nums[right])
                {
                    left = mid + 1;
                }
                else    // 左半区
                {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
};
```
