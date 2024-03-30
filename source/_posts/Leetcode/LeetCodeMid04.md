---
title: LeetCode Mid(4)
tags:
  - LeetCode
  - Mid
author: fbichao
categories:
  - leetcode
  - Mid
excerpt: LeetCode Mid(4)
math: true
date: 2024-03-17 21:45:00
---
## [139. 单词拆分](https://leetcode.cn/problems/word-break/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 s 则返回 true。

> 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

- 动规、完全背包
- 时间复杂度 $O(n^2)$
- 空间复杂度 $O(n)$

```
1. 定义 dp[j] 表示字符串 s 的前 j 个字母是否出现在 wordDict 中，若 true，则只需要看剩余元素 s[j:i]
2. 状态转移 dp[i] = dp[j] && wordDict 包含 s[j:i]
3. 初始化 dp[0] = true
```

```C++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        // 构建集合
        unordered_set<string> uset(wordDict.begin(), wordDict.end());
        vector<bool> dp(s.size()+1, false);
        dp[0] = true;

        // 遍历物品
        for (int i = 1; i <= s.size(); ++i)
        {
            // 遍历背包
            for (int j = 0; j < i; ++j)
            {
                // 背包中的是单词
                string word = s.substr(j, i-j);
                if (uset.find(word) != uset.end() && dp[j])
                {
                    dp[i] = true;
                }
            }
        }

        return dp[s.size()];
    }
};
```

## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

> 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

> 不允许修改 链表。

```
![](https://file.fbichao.top/2024/03/762ab93fb1f80317cecd225f2c7d4809.png)
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

- 快慢双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
先让快指针每次多走一步，直到快慢相遇，此时从相遇点开始和从起点开始继续，再相遇就是入口
```

```C++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* slow = head;
        ListNode* fast = head;

        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;

            if (slow == fast)
            {
                ListNode* index1 = head;
                ListNode* index2 = fast;
                while (index1 != index2)
                {
                    index1 = index1->next;
                    index2 = index2->next;
                }
                return index1;
            }
        }

        return nullptr;
    }
};
```

## [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。

> 实现 LRUCache 类：
> LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
> int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
> void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。

> 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

```
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

- 双向链表（实现快速增删）+哈希表（实现快速查找修改）
- 时间复杂度 $O(1)$
- 空间复杂度 $O(n)$

```
- get
    首先在 dict 中查找，找不到 return -1；找到了，插入到 cache 头部，删除旧的，更新 dict

- put
    首先在 dict 中查找，找到了，从 cache 和 dict 中删除旧的，加入新的，是否超过 capacity
```

```C++
class LRUCache {
private:
    const int m_capacity;
    list<pair<int, int>> cache; // 存储 key-value
    unordered_map<int, decltype(cache.begin())> dict; // key 和 key 在链表中的位置
  
public:
    // const 需要列表初始化
    LRUCache(int capacity) : m_capacity(capacity)
    { }
  
    int get(int key) {
        // 在 dict 中 O(1) 查找
        auto it = dict.find(key);
        if (it == dict.end()) return -1;

        // 放到链表头部，最近访问
        cache.push_front(*it->second);
        // 删除旧的
        cache.erase(it->second);
        // 更新 dict 的指向
        dict[key] = cache.begin();

        return cache.front().second;
    }
  
    void put(int key, int value) {
        auto it = dict.find(key);
        if (it != dict.end())
        {
            cache.erase(it->second);
            dict.erase(it);
        }

        cache.push_front({key, value});
        dict.insert({key, cache.begin()});

        if (cache.size() > m_capacity)
        {
            dict.erase(cache.back().first);
            cache.pop_back();
        }
    }
};
```

## [148. 排序链表](https://leetcode.cn/problems/sort-list/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

```
![](https://file.fbichao.top/2024/03/3063f434862388174e067a742741af12.png)
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

- 归并排序
- 时间复杂度 $O(n\cdot{logn})$
- 空间复杂度 $O(1)$

```
首先使用快慢双指针拆分链表，然后对两个有序链表进行合并
```

```C++
class Solution {
public:
    ListNode* merge(ListNode* left, ListNode* right)
    {
        // 归并两个有序链表
        ListNode* dummpy = new ListNode(-1);
        ListNode* cur = dummpy;
      
        while (left && right)
        {
            if (left->val > right->val)
            {
                cur->next = right;
                right = right->next;
            }
            else
            {
                cur->next = left;
                left = left->next;
            }
            cur = cur->next;
        }

        cur->next = left == nullptr ? right:left;
        return dummpy->next;
    }

    ListNode* mergeSort(ListNode* head)
    {
        // base case
        if (head == nullptr || head->next == nullptr) return head;

        // 找分割点
        ListNode* slow = head, *fast = head->next;

        while (fast && fast->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* leftNode = head;
        ListNode* rightNode = slow->next;
        slow->next = nullptr;   // 不能少

        return merge(mergeSort(leftNode), mergeSort(rightNode));
    }

    ListNode* sortList(ListNode* head) {
        return mergeSort(head);
    }
};
```

## [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/description/)

> 给你一个字符串数组 tokens ，表示一个根据 逆波兰表示法 表示的算术表达式。

> 请你计算该表达式。返回一个表示表达式值的整数。

> 注意：

> 有效的算符为 '+'、'-'、'*' 和 '/' 。
> 每个操作数（运算对象）都可以是一个整数或者另一个表达式。
> 两个整数之间的除法总是 向零截断 。
> 表达式中不含除零运算。
> 输入是一个根据逆波兰表示法表示的算术表达式。
> 答案及所有中间计算结果可以用 32 位 整数表示。

```
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```

- 栈模拟
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
如果是符号，就取出栈中元素计算后入栈，注意计算的顺序
如果是数字，转化为 int 入栈
```

```C++
class Solution {
public:
    int str2int(string& str)
    {
        int index = 0;
        // 考虑正负号
        if (str[0] == '-') index = 1;

        int num = 0;
        while (index < str.size())
        {
            num = num * 10 + (str[index] - '0');
            ++index;
        }

        return str[0] == '-' ? -num: num;
    }

    int evalRPN(vector<string>& tokens) {
        stack<int> st;

        for (auto str: tokens)
        {
            // 不是符号入栈，转化为 int
            if (str != "+" && str != "-" && str != "*" && str != "/")
            {
                st.push(str2int(str));
            }
            // 否则根据符号计算
            else
            {
                auto n1 = st.top(); st.pop();
                auto n2 = st.top(); st.pop();
                if (str == "+") st.push(n2 + n1);
                if (str == "-") st.push(n2 - n1);
                if (str == "*") st.push(n2 * n1);
                if (str == "/") st.push(n2 / n1);
            }
        }

        return st.top();
    }
};
```



## [151. 反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/description/)

> 给你一个字符串 s ，请你反转字符串中 单词 的顺序。

> 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

> 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。

> 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

```
输入：s = "  hello world  "
输出："world hello"
解释：反转后的字符串中不能存在前导空格和尾随空格。
```

- 双指针
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
1. 反转整个字符串
2. 双指针删除多余空格
3. 双指针反转每个单词
4. 双指针反转最后一个单词
```


```C++
class Solution {
public:
    string reverseWords(string s) {
        // 先反转整个字符串 O(n)
        reverse(s.begin(), s.end());

        // 通过快慢双指针，删除多余空格
        int slow = 0, fast = 0;
        while (fast < s.size())
        {   // 找到首个非空字符
            if (s[fast] != ' ')
            {
                // 在下次遍历的单词前加上空格
                if (slow != 0) s[slow++] = ' ';

                // 直到遇到空格停止
                while (fast < s.size() && s[fast] != ' ')
                {
                    s[slow++] = s[fast++];
                }
            }
            ++fast;
        }
        
        // resize
        s.resize(slow);
        
        // 双指针反转每个单词
        int left = 0, right = 0;
        while (right < s.size())
        {
            if (s[right] == ' ')
            {
                reverse(s.begin()+left, s.begin()+right);
                left = right + 1;
            }
            ++right;
        }

        // 反转最后一个单词
        reverse(s.begin() + left, s.end());
        return s;
    }
};
```


## [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

> 测试用例的答案是一个 32-位 整数。

```
输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

- 动规
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
由于负数乘以负数得到正数，需要同时考虑最小和最大
dp_min[i] 以 nums[i] 结尾的最小乘积
dp_max[i] 以 nums[i] 结尾的最大乘积
```

```C++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp_min(n, 0);
        vector<int> dp_max(n, 0);

        dp_min[0] = nums[0];
        dp_max[0] = nums[0];
        int res = nums[0];

        for (int i = 1; i < n; ++i)
        {
            dp_min[i] = min({dp_min[i-1] * nums[i], nums[i], dp_max[i-1]*nums[i]});
            dp_max[i] = max({dp_min[i-1] * nums[i], nums[i], dp_max[i-1]*nums[i]});

            res = max(res, dp_max[i]);
        }
        return res;
    }
};
```

## [155. 最小栈]()

> 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

> 实现 MinStack 类:

> MinStack() 初始化堆栈对象。
> void push(int val) 将元素val推入堆栈。
> void pop() 删除堆栈顶部的元素。
> int top() 获取堆栈顶部的元素。
> int getMin() 获取堆栈中的最小元素。

```
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.
```

- 两个栈
- 时间复杂度 $O(1)$
- 空间复杂度 $O(n)$

```

```

```C++
class MinStack {
private:
    // 一个正常存储数据
    stack<int> st;
    // 一个存储当前最小值
    stack<int> min_st;

public:
    MinStack() {
        // 先加入一个 MAX
        min_st.push(INT_MAX);
    }
  
    void push(int val) {
        st.push(val);
        // 每次有新的 val 加入，比较一下，看当前的最小值是什么
        min_st.push(min(min_st.top(), val));
    }
  
    void pop() {
        // 一起 pop
        st.pop();
        min_st.pop();
    }
  
    int top() {
        return st.top();
    }
  
    int getMin() {
        return min_st.top();
    }
};
```

## [198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

> 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

- 动规
- 时间复杂度 $O(n)$
- 空间复杂度 $O(n)$

```
1. dp[i] 定义为偷至第 i 家时候的最大值
2. dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
3. 初始化两个即可
```

```C++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();

        if (n == 0) return 0;
        if (n == 1) return nums[0];

        vector<int> dp(n, 0);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);

        for (int i = 2; i < n; ++i)
        {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }

        return dp[n-1];
    }
};
```

## [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

> 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

> 此外，你可以假设该网格的四条边均被水包围。

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

- DFS/BFS
- 时间复杂度 $O(mn)$
- 空间复杂度 $O(mn)$

```
是否允许原地修改，不允许需要额外使用 visited
```

```C++
class Solution {
public:
    void dfs(vector<vector<char>>& grid, vector<vector<bool>>& visited, int row, int col)
    {
        int m = grid.size();
        int n = grid[0].size();

        // 先判断 row 和 col
        if (row < 0 || row >= m || col < 0 || col >= n) return;
        if (grid[row][col] == '0') return;
        if (visited[row][col]) return;

        visited[row][col] = true;

        dfs(grid, visited, row + 1, col);
        dfs(grid, visited, row, col + 1);
        dfs(grid, visited, row - 1, col);
        dfs(grid, visited, row, col - 1);
    }

    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
        for (int i = 0; i < grid.size(); ++i)
        {
            for (int j = 0; j < grid[0].size(); ++j)
            {
                if (grid[i][j] == '1' && !visited[i][j])
                {
                    dfs(grid, visited, i, j);
                    count++;
                }
            }
        }
        return count;
    }
};
```

## [207. 课程表](https://leetcode.cn/problems/course-schedule/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

> 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

> 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。

> 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

```
输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
输出：false
解释：总共有 2 门课程。学习课程 1 之前，你需要先完成课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
```

- BFS
- 时间复杂度 $O(m+n)$，其中 $n$ 为课程数，$m$ 为先修课程的要求数。这其实就是对图进行广度优先搜索的时间复杂度。
- 空间复杂度 $O(m+n)$

```
先构造 {先修课程: [后修课程……]}
再构造每个课程的入度 {课程号: 入度}，即表示该课程需要先修的课程数
将入度为 0 的节点入队列，遍历他们的后修课程
```

```C++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> edges(numCourses, vector<int>(0));
        vector<int> indeg(numCourses, 0);

        for (auto info: prerequisites)
        {
            // 先修: [后修]
            edges[info[1]].push_back(info[0]);
            // 课程: 入度
            ++indeg[info[0]];
        }

        queue<int> que;
        for (int i = 0; i < numCourses; ++i)
        {   // 先将入度为 0 的入队列
            if (indeg[i] == 0)
            {
                que.push(i);
            }
        }

        int visited = 0;
        while (!que.empty())
        {
            int u = que.front(); que.pop();
            // 每次遍历一个节点就加一次
            ++visited;

            // 遍历 u 的后续课程
            for (int v: edges[u])
            {
                // 减少后续课程的入度
                --indeg[v];
                // 如果为 0，也加入队列
                if (indeg[v] == 0)
                {
                    que.push(v);
                }
            }
        }

        return visited == numCourses;
    }
};
```

## [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=featured-list&envId=2cktkvj?envType=featured-list&envId=2cktkvj)

> Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

> 请你实现 Trie 类：

> Trie() 初始化前缀树对象。

> void insert(String word) 向前缀树中插入字符串 word 。
> boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
> boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

- 
- 时间复杂度 $O()$
- 空间复杂度 $O()$

```

```

```C++



```


## [209. 长度最小的子数组]()

> 给定一个含有 n 个正整数的数组和一个正整数 target 。

> 找出该数组中满足其总和大于等于 target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。


```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

- 滑动窗口
- 时间复杂度 $O(n)$
- 空间复杂度 $O(1)$

```
滑动窗口，维护满足条件的范围
```


```C++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        // 滑动窗口
        int slow = 0, fast = 0;

        int sum = 0;
        int res = nums.size() + 1;

        while (fast < nums.size())
        {
            // 求窗口内的和
            sum += nums[fast];
            while (sum >= target)
            {   // 如果和大于满足条件
                res = min(res, fast - slow + 1);
                sum -= nums[slow++];
            }
            ++fast;
        }

        return res == nums.size() + 1 ? 0 : res;
    }
};
```


## [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/description/)

> 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

> 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

```
输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
```

```C++
class Solution {
public:
    int robbyRange(vector<int>& nums, int left, int right)
    {
        if (left == right) return nums[left];

        vector<int> dp(nums.size(), 0);
        dp[left] = nums[left];
        dp[left+1] = max(nums[left], nums[left+1]);

        for (int i = left + 2; i <= right; ++i)
        {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        }

        return dp[right];
    }

    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return 0;
        if (n == 1) return nums[0];
        
        // 不偷最后一家和不偷第一家分别讨论
        return max(robbyRange(nums, 0, n - 2), robbyRange(nums, 1, n-1));
    }
};
```