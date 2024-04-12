---
title: codetop中等(4)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Codetop
excerpt: codetop中等(4)
math: true
date: 2024-04-07 21:45:00
---

## [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/submissions/520722933/)

- 前缀和

```C++
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> umap;   // 存储前缀和
        // 当 preSum[i-1] 和 k 相等时
        umap[0] = 1;

        int preSum = 0;
        int ans = 0;

        // 从 i~j 的和为 preSum[j] - preSum[i-1]
        for (int i = 0; i < nums.size(); ++i)
        {
            preSum += nums[i];
            // preSum[j] - preSum[i-1] = k
            // preSum[i-1] = preSum[j] - k
            if (umap.count(preSum - k))
            {
                // 加上次数
                ans += umap[preSum - k];
            }
            // 存入 preSum[i] 的值
            umap[preSum]++;
        }

        return ans;
    }
};
```


## [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/)

- 二分搜索

```C++
class Solution {
public:
    int findMin(vector<int>& nums) {
        // 分成两部分
        // 左半边最小的都比右半边最大的大
        int left = 0, right = nums.size() - 1;
        while (left < right)
        {
            int mid = left + (right - left) / 2;
            // 需要右移动
            if (nums[mid] > nums[right])
            {
                left = mid + 1;
            }
            else    // 可能 mid 就是结果
            {
                right = mid;
            }
        }
        return nums[left];
    }
};
```


## [468. 验证IP地址](https://leetcode.cn/problems/validate-ip-address/description/)

- 模拟

```C++
class Solution {
public:
    string validIPAddress(string queryIP) {
        // prev 和 cur 记录两个.之间的字符串
        // 开头 prev = -1
        // 结尾 cur = size()
        if (queryIP.find('.') != string::npos)
        {
            int prev = -1;
            for (int i = 0; i < 4; ++i)
            {
                // 尾部
                int cur = (i == 3 ? queryIP.size() : queryIP.find('.', prev+1));
                if (cur == string::npos)
                {
                    return "Neither";
                }
                // 长度不满足
                if (cur - prev - 1 < 1 || cur - prev - 1 > 3)
                {
                    return "Neither";
                }
                int addr = 0;
                for (int j = prev + 1; j < cur; ++j)
                {
                    if (!isdigit(queryIP[j]))   // 不是数字
                    {
                        return "Neither";
                    }
                    addr = addr * 10 + (queryIP[j] - '0');
                }
                // 大于 255
                if (addr > 255) return "Neither";
                // 前缀 0，且数字大于 0
                if (addr > 0 && queryIP[prev + 1] == '0') return "Neither";
                // 前缀 0 且数字等于 0
                if (addr == 0 && cur - prev - 1 > 1) return "Neither";
                prev = cur;
            }
            return "IPv4";
        }
        else
        {
            int prev = -1;
            for (int i = 0; i < 8; ++i)
            {
                int cur = (i == 7 ? queryIP.size() : queryIP.find(':', prev+1));
                if (cur == string::npos) return "Neither";
                if (cur - prev - 1 < 1 || cur - prev - 1 > 4) return "Neither";
                for (int j = prev + 1; j < cur; ++j)
                {
                    if (!isdigit(queryIP[j]) && !('a' <= tolower(queryIP[j]) && tolower(queryIP[j]) <= 'f'))
                    {
                        return "Neither";
                    }
                }
                prev = cur;
            }
            return "IPv6";
        }
    }
};
```


## [138. 随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/description/)

- 哈希表存储旧的和新的节点

```C++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node*, Node*> umap;

        Node* cur = head;
        while (cur)
        {
            umap[cur] = new Node(cur->val);
            cur = cur->next;
        }

        cur = head;
        while (cur)
        {
            if (cur->next)
                umap[cur]->next = umap[cur->next];
            if (cur->random)
                umap[cur]->random = umap[cur->random];
            
            cur = cur->next;
        }

        return umap[head];
    }
};
```



## [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/description/)

- 回溯

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& nums, vector<bool>& used)
    {
        if (nums.size() == path.size()) 
        {
            res.push_back(path);
            return;
        }

        // for 循环是选择结果中的每个构成元素
        // backtrack 则是根据 for 的元素，构成完整的结果
        for (int i = 0; i < nums.size(); ++i)
        {
            if (i > 0 && nums[i] == nums[i-1] && used[i-1] == false) continue;

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

    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<bool> used(nums.size(), false);
        backtrack(nums, used);
        return res;
    }
};
```


## [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/description/)

- 单调栈

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


## [207. 课程表](https://leetcode.cn/problems/course-schedule/description/)

- 拓扑排序

```C++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> indegree(numCourses, 0);      // 每个节点的入度
        vector<vector<int>> graph(numCourses, vector<int>(0)); // 构建邻接表，节点:[节点 1, 节点 2]

        // 构建图和入度表
        for (auto info: prerequisites)
        {
            graph[info[1]].push_back(info[0]);
            ++indegree[info[0]];
        }

        // 度为 0 的节点入队列
        queue<int> que;
        for (int i = 0; i < numCourses; ++i)
        {
            if (indegree[i] == 0)
            {
                que.push(i);
            }
        }

        while (!que.empty())
        {
            // 取出一个节点
            auto node = que.front(); que.pop();
            // 对邻居遍历，并减少入度
            for (auto val: graph[node])
            {
                indegree[val]--;
                if (indegree[val] == 0)
                {
                    que.push(val);
                }
            }
        }

        // 发生变化的只有 indegree
        for (auto val: indegree)
        {
            if (val) return false;
        }
        return true;
    }
};
```


## [498. 对角线遍历](https://leetcode.cn/problems/diagonal-traverse/description/)

- 模拟

```C++
class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
        // 观察行和列坐标发现
        // 向右上方动，和为偶数
        // 向左下方动，和为奇数

        int m = mat.size(), n = mat[0].size();

        vector<int> res;
        int loop = m * n;
        int row = 0, col = 0;
        while (loop--)
        {
            res.push_back(mat[row][col]);

            if ((row + col) % 2)    // 奇数，向左下，特殊 case 要么碰到左边界，要么下边界
            {   
                // if else if 先后顺序有讲究
                // 考虑在左下角时怎么移动，谁就先判断
                if (row == m - 1)  // 是第一列最后一行
                {
                    col++;  // 增加列
                }
                else if (col == 0)   // 如果是第一列，并且不是第一列最后一个元素
                {
                    row++;  // 增加行数
                }
                else    // 否则就向左下角移动，增加行，减少列
                {
                    ++row;
                    --col;
                }
            }
            else    // 偶数，右上移动，特殊 case 要么上边界，要么右边界
            {
                if (col == n - 1)   // 右边界，加行
                {
                    ++row;
                }
                else if (row == 0)  // 上边界，加列
                {
                    ++col;
                }
                else    // 其余情况
                {
                    --row;
                    ++col;
                }
            }
        }
        return res;
    }
};
```

## [402. 移掉 K 位数字](https://leetcode.cn/problems/remove-k-digits/description/)

- 单调栈

```C++
class Solution {
public:
    string removeKdigits(string num, int k) {
        int n = num.size();
        if (n <= k) return "0";

        stack<int> st;
        for (int i = 0; i < n; ++i)
        {
            while (!st.empty() && num[i] < num[st.top()] && k)
            {
                int ind = st.top(); st.pop();
                --k;
            }
            st.push(i);
        }

        while (k--)
        {
            st.top();
            st.pop();
        }

        string ans = "";
        while (!st.empty())
        {
            ans += num[st.top()]; st.pop();
        }
        reverse(ans.begin(), ans.end());
        int i;
        for (i = 0; i < ans.size(); ++i)
        {
            if (ans[i] != '0')
            {
                break;
            }
        }
        ans = ans.substr(i);
        return ans == "" ? "0" : ans;   // 存在 100，1 case
    }
};
```


## [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)

- 双指针，以较短的边算过面积之后，移动短边

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
            // 移动最短边，因为最短边不动，不可能构成更大的 res
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


## [LCR 155. 将二叉搜索树转化为排序的双向链表](https://leetcode.cn/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/description/)

- 利用中序遍历是递增的

```C++
class Solution {
private:
    Node* pre = nullptr;
    Node* head = nullptr;

public:
    void dfs(Node* cur)
    {
        if (cur == nullptr) return;
        dfs(cur->left);
        if (pre != nullptr) pre->right = cur;
        else head = cur;
        cur->left = pre;
        pre = cur;
        dfs(cur->right);
    }

    Node* treeToDoublyList(Node* root) {
        if (root == nullptr) return nullptr;
        dfs(root);
        head->left = pre;
        pre->right = head;
        return head;
    }
};
```


## [堆排序](codetop中等(1).md#912-排序数组)


## [958. 二叉树的完全性检验](https://leetcode.cn/problems/check-completeness-of-a-binary-tree/description/)

- 层序遍历，出现一个空节点后，不允许出现非空节点

```C++
class Solution {
public:
    // 出现空节点之后，不能再出现非空节点
    bool isCompleteTree(TreeNode* root) {
        queue<TreeNode*> que;
        que.push(root);
        bool flag = false;

        while (!que.empty())
        {
            auto node = que.front();
            que.pop();

            if (node == nullptr)
            {
                flag = true;
                continue;
            }

            // 如果 node 非空，并且之前有空节点，则 false
            if (flag) return false;
            que.push(node->left);
            que.push(node->right);
        }

        return true;
    }
};
```


## [排序奇升偶降链表](https://mp.weixin.qq.com/s/0WVa2wIAeG0nYnVndZiEXQ)

一个链表，奇数位置升序，偶数位置降序，进行从小到大排序
1. 按奇偶位置拆分链表，得1->3->5->7->NULL和8->6->4->2->NULL
2. 反转偶链表，得1->3->5->7->NULL和2->4->6->8->NULL
3. 合并两个有序链表，得1->2->3->4->5->6->7->8->NULL


## [检测循环依赖](https://mp.weixin.qq.com/s/pCRscwKqQdYYN7M1Sia7xA)

- 拓扑排序


## [79. 单词搜索](https://leetcode.cn/problems/word-search/description/)

- 回溯，dfs

```C++
class Solution {
public:
    bool dfs(vector<vector<char>>& board, string word, int row, int col, int index)
    {
        if (index == word.size())
        {
            return true;
        }
        if (row < 0 || row >= board.size()) return false;
        if (col < 0 || col >= board[0].size()) return false;
        if (board[row][col] != word[index]) return false;

        board[row][col] = '\0'; // 回溯，标记

        // 上下左右搜素
        bool res = dfs(board, word, row+1, col, index + 1)
                    || dfs(board, word, row-1, col, index + 1)
                    || dfs(board, word, row, col+1, index + 1)
                    || dfs(board, word, row, col-1, index + 1);
        
        board[row][col] = word[index];      // 回溯，恢复

        return res;
    }

    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size();
        int n = board[0].size();

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }
};
```


## [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/description/)

- 贪心，记录一路走来，可以跳的最远位置

```C++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // maxVal 记录当前可以调到的最远地方
        int maxVal = 0;
        for (int i = 0; i < nums.size(); ++i)
        {
            // 如果当前位置超过了最远地方，false
            if (i > maxVal) return false;
            // 最远地方超过了数组长度，true
            if (maxVal >= nums.size() - 1) return true;
            // 更新
            maxVal = max(maxVal, i + nums[i]);
        }
        return true;
    }
};
```


## [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/description/)

- 含有重复元素的去重组合

```C++
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;

public:
    void backtrack(vector<int>& candidates, int target, vector<bool>& used, int sum, int index)
    {
        if (sum == target)
        {
            res.push_back(path);
            return;
        }
        if (sum > target) return;

        // 开始的地方，会变化，所以从 index 开始
        for (int i = index; i < candidates.size(); ++i)
        {
            // 1 1 2 5
            // 如果在使用第二个 1 时，前一个 1 没使用，则 continue
            if (i > 0 && candidates[i] == candidates[i-1] && used[i-1] == false)
            {
                continue;
            }

            used[i] = true;
            path.push_back(candidates[i]);
            sum += candidates[i];
            // 下一次遍历的值，需要 i+1 不可重复选
            backtrack(candidates, target, used, sum, i + 1);
            sum -= candidates[i];
            path.pop_back();
            used[i] = false;
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        // 需要去重，一般都要排序
        sort(candidates.begin(), candidates.end());
        vector<bool> used(candidates.size(), false);
        backtrack(candidates, target, used, 0, 0);
        return res;
    }
};
```


## [74. 搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/description/)

- 从右上角开始搜索

```C++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = 0, col = matrix[0].size() - 1;

        while (row < matrix.size() && col >= 0)
        {
            if (matrix[row][col] > target)
            {
                --col;
            }
            else if (matrix[row][col] < target)
            {
                ++row;
            }
            else
            {
                return true;
            }
        }

        return false;
    }
};
```


## [61. 旋转链表](https://leetcode.cn/problems/rotate-list/description/)

- 遍历链表得到长度，取模后，找到旋转后的尾结点，断开

```C++
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (head == nullptr) return nullptr;
        int len = 0;
        ListNode* cur = head;
        ListNode* end;     // 链表最后一个节点
        while (cur)
        {
            ++len;
            end = cur;
            cur = cur->next;
        }

        k %= len;   // 需要移动几次

        ListNode* split = head;
        int split_index = len - k - 1;
        while (split_index--)
        {
            split = split->next;    // 新的链表的尾结点
        }

        end->next = head;   // 旧的链表首尾相连
        // 保存 split 的下一个节点，作为新的起始节点
        ListNode* newHead = split->next;
        split->next = nullptr;

        return newHead;
    }
};
```


























































