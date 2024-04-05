---
title: codetop中等(1)
tags:
  - LeetCode
  - codetop 中等
author: fbichao
categories:
  - leetcode
  - Easy
excerpt: codetop中等(1)
math: true
date: 2024-04-01 21:45:00
---

## [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

- 滑动窗口

```C++
class Solution {
public:
    // 滑动窗口
    int lengthOfLongestSubstring(string s) {
        unordered_map<int, int> umap;

        int slow = 0, fast = 0;
        int res = 0;
        while (fast < s.size())
        {
            umap[s[fast]]++;
            while (umap[s[fast]] > 1)
            {
                res = max(res, fast - slow);
                umap[s[slow]]--;
                ++slow;
            }
            ++fast;
        }
        res = max(res, fast - slow);
        return res;
    }
};
```


## [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/description/)

- list + unorder_map

```C++
class LRUCache {
private:
    // 存储 key-value
    list<pair<int, int>> cache;
    // key 和 key 在链表中的位置
    unordered_map<int, decltype(cache.begin())> umap;
    // 容量
    const int m_capacity;

public:
    // const 需要列表初始化
    LRUCache(int capacity) : m_capacity(capacity)
    { }
    
    int get(int key) {
        pair<int, int> p = {key, -1};
        auto it = umap.find(key);
        
        // 找不到 key
        if (it == umap.end()) return -1;

        // 找到 key，修改 value
        p.second = it->second->second;
        cache.push_front(p);
        // 删除旧的
        cache.erase(it->second);
        // 更新 dict 的指向
        umap[key] = cache.begin();

        return cache.front().second;
    }
    
    void put(int key, int value) {
        pair<int, int> p = {key, value};
        auto it = umap.find(key);

        // 查询到了 key，删除
        if (it != umap.end())
        {
            cache.erase(it->second);
        }

        // 加入新的 pair
        cache.push_front(p);
        umap[key] = cache.begin();

        if (cache.size() > m_capacity)
        {
            umap.erase(cache.back().first);
            cache.pop_back();
        }
    }
};
```

- 自己实现双向链表

```C++
struct DoubleList
{
    int key;
    int value;
    DoubleList* prev;
    DoubleList* next;
    DoubleList(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DoubleList(int k, int v) : key(k), value(v) {}
};

class LRUCache {
private:
    DoubleList* head;
    DoubleList* tail;
    // key 和 key 在链表中的位置
    unordered_map<int, DoubleList*> umap;
    // 容量
    const int m_capacity;
    int size = 0;

public:
    // const 需要列表初始化
    LRUCache(int capacity) : m_capacity(capacity)
    {
        // 虚拟头和虚拟尾
        head = new DoubleList();
        tail = new DoubleList();
        head->next = tail;
        tail->prev = head;
    }
    
    int get(int key) {
        auto it = umap.find(key);
        
        // 找不到 key
        if (it == umap.end()) return -1;

        // 找到 key，修改 value
        DoubleList* node = umap[key];
        moveToHead(node);
        return node->value;
    }
    
    void put(int key, int value) {
        if (!umap.count(key))
        {
            DoubleList* node = new DoubleList(key, value);
            umap[key] = node;
            addToHead(node);
            ++size;
            if (size > m_capacity)  // 主需要删除一次
            {
                DoubleList* remove = removeTail();
                umap.erase(remove->key);
                delete remove;
                --size;
            }
        }
        else
        {
            DoubleList* node = umap[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DoubleList* node)
    {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }

    void removeNode(DoubleList* node)
    {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DoubleList* node)
    {
        removeNode(node);
        addToHead(node);
    }

    DoubleList* removeTail()
    {
        DoubleList* node = tail->prev;
        removeNode(node);
        return node;
    }
};
```

## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/)

- 优先队列

```C++
class Solution {
public:
    struct cmp
    {
        bool operator()(const int a, const int b)
        {
            return a < b;
        }
    };
    
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, cmp> pri_que;        

        for (int i = 0; i < nums.size(); ++i)
        {
            pri_que.push(nums[i]);
        }

        for (int i = 0; i < k - 1; ++i)
        {
            pri_que.pop();
        }

        return pri_que.top();
    }
};
```

- 快速选择算法

```C++
class Solution {
public:
    // 分
    int quickSelect(vector<int>& nums, int k, int left, int right)
    {
        if (left == right) return nums[left];
        int split = quickSort(nums, left, right);
        if (split + 1 == k) return nums[split];
        else if (split + 1 < k) return quickSelect(nums, k, split + 1, right);
        else return quickSelect(nums, k, left, split - 1);
    }

    // 治
    int quickSort(vector<int>& nums, int left, int right)
    {
        int base = nums[right];
        // 左大右小
        int slow = left, fast = left;
        while (fast < right)
        {
            if (nums[fast] > base)
            {
                swap(nums[fast], nums[slow]);
                ++slow;
            }
            ++fast;
        }
        swap(nums[slow], nums[right]);
        return slow;
    }

    int findKthLargest(vector<int>& nums, int k) {
        int res = quickSelect(nums, k, 0, nums.size() - 1);
        return res;
    }
};
```


## [15. 三数之和](https://leetcode.cn/problems/3sum/description/)

- 排序 + 三指针

```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        // 首先排序
        sort(nums.begin(), nums.end());

        // 三个指针
        vector<vector<int>> res;
        for (int first = 0; first < nums.size(); ++first)
        {
            if (first > 0 && nums[first] == nums[first - 1]) continue;

            int second = first + 1;
            int third = nums.size() - 1;
            while (second < third)
            {
                int sum = nums[first] + nums[second] + nums[third];
                if (sum > 0) --third;
                else if (sum < 0) ++second;
                else if (sum == 0)
                {
                    res.push_back({nums[first], nums[second], nums[third]});

                    while (second < third && nums[second] == nums[second + 1]) ++second;
                    while (second < third && nums[third] == nums[third - 1]) --third;
                    ++second;
                    --third;
                }
            }
        }

        return res;
    }
};
```

## [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

- 动态规划，题目要求是连续的子数组，所以必须是 dp(n)

```C++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        // 题目要求是连续的子数组，所以必须是 dp(n)
        // 即 dp[i] 表示以 nums[i] 结尾的最大连续子数组之和
        int n = nums.size();
        vector<int> dp(n, 0);

        dp[0] = nums[0];

        int res = dp[0];    // 可能在中间段产生结果

        for (int i = 1; i < n; ++i)
        {
            // 必须加上 nums[i]，单独或者合并
            dp[i] = max(dp[i-1]+nums[i], nums[i]);

            res = res < dp[i] ? dp[i] : res;
        }

        return res;
    }
};
```

## [912. 排序数组](https://leetcode.cn/problems/sort-an-array/description/)

- 快排

```C++
class Solution {
public:
    void randomQuickSort(vector<int>& nums, int left, int right)
    {
        if (left < right)
        {
            int pos = quickSort(nums, left, right);
            randomQuickSort(nums, left, pos - 1);
            randomQuickSort(nums, pos + 1, right);
        }
    }

    int quickSort(vector<int>& nums, int left, int right)
    {
        int index = left + random() % (right - left + 1);
        swap(nums[index], nums[right]);
        int base = right;

        int slow, fast;
        for (slow = left, fast = left; fast < base; ++fast)
        {
            if (nums[fast] < nums[base])
            {
                swap(nums[slow], nums[fast]);
                ++slow;
            }
        }
        swap(nums[slow], nums[base]);
        return slow;
    }

    vector<int> sortArray(vector<int>& nums) {
        srand((unsigned)time(NULL));
        randomQuickSort(nums, 0, nums.size() - 1);
        return nums;
    }
};
```

- 归并

```C++
class Solution {
public:
    vector<int> merge(vector<int>& nums1, vector<int>& nums2)
    {
        vector<int> res;

        int n1, n2;
        for (n1 = 0, n2 = 0; n1 < nums1.size() && n2 < nums2.size();)
        {
            if (nums1[n1] < nums2[n2])
            {
                res.push_back(nums1[n1]);
                ++n1;
            }
            else
            {
                res.push_back(nums2[n2]);
                ++n2;
            }
        }

        while (n2 < nums2.size()) res.push_back(nums2[n2++]);

        while (n1 < nums1.size()) res.push_back(nums1[n1++]);

        return res;
    }

    vector<int> mergeSort(vector<int>& nums, int left, int right)
    {
        if (left == right) return {nums[left]};
        int mid = left + (right - left) / 2;
        vector<int> leftMerge = mergeSort(nums, left, mid);
        vector<int> rightMerge = mergeSort(nums, mid + 1, right);
        return merge(leftMerge, rightMerge);
    }

    vector<int> sortArray(vector<int>& nums) {
        return mergeSort(nums, 0, nums.size() - 1);
    }
};
```

- 堆排序

```C++
class Solution {
public:
    void buildMaxHeap(vector<int>& nums)
    {
        int n = nums.size();
        for (int i = n / 2; i >= 0; --i)
        {
            maxHeapify(nums, i, n);
        }
    }

    void maxHeapify(vector<int>& nums, int i, int n)
    {
        while ((i * 2 + 1) < n)
        {
            int lSon = 2 * i + 1;
            int rSon = 2 * i + 2;
            int large = i;

            if (lSon < n && nums[lSon] > nums[i]) large = lSon;
            if (rSon < n && nums[rSon] > nums[large]) large = rSon;

            if (large != i)
            {
                swap(nums[i], nums[large]);
                i = large;
            }
            else break;
        }
    }

    vector<int> sortArray(vector<int>& nums) {
        int n = nums.size();
        buildMaxHeap(nums);

        for (int i = n - 1; i >= 1; --i)
        {
            swap(nums[0], nums[i]);
            --n;
            maxHeapify(nums, 0, n);
        }

        return nums;
    }
};
```


## [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/)

- 动态规划，注意变量方向

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

        // 观察 dp[i][j] = dp[i+1][j-1]
        // 所以 i 的遍历方向是从右往左
        // 所以 j 的遍历方向是从左往右
        for (int i = n - 1; i >= 0; --i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                // 两边相等
                if (s[i] == s[j])
                {   // 直接判断
                    if (j - i <= 2) dp[i][j] = true;
                    else    // 中心扩展
                    {
                        dp[i][j] = dp[i+1][j-1];
                    }
                }

                // 更新
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



## [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/)

- 二分搜索，分两个区间段讨论

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


## [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/)

- 层序遍历

```C++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if (root == nullptr) return {};

        queue<TreeNode*> que;
        que.push(root);
        
        vector<vector<int>> res;
        while (!que.empty())
        {
            int n = que.size();

            vector<int> temp;
            for (int i = 0; i < n; ++i)
            {
                auto node = que.front(); que.pop();
                temp.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            res.push_back(temp);
        }

        return res;
    }
};
```


## [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/description/)

- 深度优先搜索

```C++
class Solution {
public:
    void dfs(vector<vector<char>>& grid, vector<vector<bool>>& visited, int row, int col)
    {
        int m = grid.size();
        int n = grid[0].size();

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


## [46. 全排列](https://leetcode.cn/problems/permutations/description/)

- 回溯

```C++
// 回溯
class Solution {
private:
    vector<int> path;
    vector<vector<int>> res;

public:
    void dfs(vector<int>& nums, vector<bool>& visited)
    {
        if (path.size() == nums.size())
        {
            res.push_back(path);
            return;
        }

        // 遍历每个元素
        for (int i = 0; i < nums.size(); ++i)
        {
            if (!visited[i])
            {
                visited[i] = true;
                path.push_back(nums[i]);
                dfs(nums, visited); // 每次都从头遍历，不需要添加 index
                path.pop_back();
                visited[i] = false;
            }
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> visited(nums.size(), false);
        dfs(nums, visited);
        return res;
    }
};
```


## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/)

- 先序遍历递归

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // ① 根节点就是 p 或 q
        // ② p 和 q 在 root 的两侧
        if (root == nullptr) return nullptr;
        if (root == p || root == q) return root;

        auto leftNode = lowestCommonAncestor(root->left, p, q);
        auto rightNode = lowestCommonAncestor(root->right, p, q);

        // p 和 q 分散在两边
        if (leftNode && rightNode) return root;
        // 在同一侧
        return leftNode == nullptr ? rightNode : leftNode;
    }
};
```


## [103. 二叉树的锯齿形层序遍历](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/)

- 层序遍历 + 奇偶情况

```C++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if (root == nullptr) return {};
        queue<TreeNode*> que;
        vector<vector<int>> res;

        que.push(root);

        int order = 0;
        while (!que.empty())
        {
            int size = que.size();

            vector<int> temp;
            for (int i = 0; i < size; ++i)
            {
                TreeNode* node = que.front(); que.pop();

                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
                temp.push_back(node->val);
            }

            if (order % 2)
            {
                reverse(temp.begin(), temp.end());
            }
            order = !order;

            res.push_back(temp);
        }

        return res;
    }
};
```


## [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii/description/)

- 链表

```C++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;

        ListNode* leftNode = dummy;
        ListNode* rightNode = dummy;

        // 区间前一个节点
        left--;
        while (left--)
        {
            leftNode = leftNode->next;
        }

        // 区间后一个节点
        right++;
        while (right--)
        {
            rightNode = rightNode->next;
        }

        // 待翻转的一个节点
        ListNode* prev = leftNode->next;
        // 区间反转之前首个节点
        ListNode* end = prev;
        // 待翻转的另一个节点
        ListNode* cur = prev->next;
        while (cur != rightNode)
        {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }
        // prev 此时只想反转前区间最后一个节点
        leftNode->next = prev;
        // end 就是反转前第一个节点
        end->next = cur;

        return dummy->next;
    }
};
```


## [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/)

- 模拟

```C++
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = n - 1;
        int top = 0, down = m - 1;
        
        vector<int> res;
        while (1)
        {
            for (int i = left; i <= right; ++i)
            {
                res.push_back(matrix[top][i]);
            }
            ++top;
            if (top > down) break;

            for (int i = top; i <= down; ++i)
            {
                res.push_back(matrix[i][right]);
            }
            --right;
            if (left > right) break;

            for (int i = right; i >= left; --i)
            {
                res.push_back(matrix[down][i]);
            }
            --down;
            if (top > down) break;

            for (int i = down; i >= top; --i)
            {
                res.push_back(matrix[i][left]);
            }
            ++left;
            if (left > right) break;
        }

        return res;
    }
};
```


## [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)

- 动态规划，两阶段

```C++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);
        int ans = 1;

        // 线性 dp，找两个点
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                if (nums[i] > nums[j])
                {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```


## [143. 重排链表](https://leetcode.cn/problems/reorder-list/description/)

- 线性表

```C++
class Solution {
public:
    void reorderList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return;
        vector<ListNode*> vec;

        ListNode* cur = head;
        while (cur)
        {
            vec.push_back(cur);
            cur = cur->next;
        }

        int left = 0, right = vec.size() - 1;
        while (left < right)
        {
            vec[left]->next = vec[right];
            ++left;
            if (left == right) break;
            vec[right]->next = vec[left];
            --right;
        }
        // 还需要将尾部置空
        vec[left]->next = nullptr;
    }
};
```


## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/)

- Floyd 判圈法

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


## [56. 合并区间](https://leetcode.cn/problems/merge-intervals/description/)

- 排序 + 合并

```C++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) return {};
        
        // 先对一个排序
        auto cmp = [](vector<int>& a, vector<int>& b)
        {
            return a[0] < b[0];
        };
        sort(intervals.begin(), intervals.end(), cmp);
        vector<vector<int>> res;

        for (int i = 0; i < intervals.size(); ++i)
        {
            int cur_L = intervals[i][0];
            int cur_R = intervals[i][1];

            // 为空，先加入
            if (res.empty()) res.push_back({cur_L, cur_R});

            // 不为空，且 res 最后一个区间右边界小于左边界，说明是新的区间
            if (!res.empty() && res.back()[1] < cur_L)
            {
                res.push_back({cur_L, cur_R});
            }
            else    // 否则合并
            {
                res.back()[1] = max(res.back()[1], cur_R);
            }
        }
        return res;
    }
};
```


## [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/)

- 快慢双指针

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
