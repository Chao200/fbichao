---
title: 高频 SQL 50 题
tags:
  - SQL
author: fbichao
categories: 
  - MySQL
excerpt: SQL 常用语法
math: false
date: 2024-04-26 21:45:00
---

[高频 SQL 50 题](https://leetcode.cn/studyplan/sql-free-50/)

## [584. 寻找用户推荐人](https://leetcode.cn/problems/find-customer-referee/description/?envType=study-plan-v2&envId=sql-free-50)

mysql 使用三个逻辑，true、false、unknown

任何与 null 值进行比较都是 unknown，包括 null

必须使用 is null 或 is not null


## [1683. 无效的推文](https://leetcode.cn/problems/invalid-tweets/description/?envType=study-plan-v2&envId=sql-free-50)

计算 varchar 字段长度

- char_length(str)
（1）计算单位：字符
（2）不管汉字还是数字或者是字母都算是一个字符

- length(str)
（1）计算单位：字节
（2）utf8编码：一个汉字三个字节，一个数字或字母一个字节。
（3）gbk编码：一个汉字两个字节，一个数字或字母一个字节。


# 2. 连接

## [left join](https://leetcode.cn/problems/replace-employee-id-with-the-unique-identifier/?envType=study-plan-v2&envId=sql-free-50)

## [left join](https://leetcode.cn/problems/product-sales-analysis-i/solutions/?envType=study-plan-v2&envId=sql-free-50)

## [left join/ group by](https://leetcode.cn/problems/customer-who-visited-but-did-not-make-any-transactions/?envType=study-plan-v2&envId=sql-free-50)

## [datediff/ join](https://leetcode.cn/problems/rising-temperature/submissions/527252727/?envType=study-plan-v2&envId=sql-free-50)

## [round/ avg/ join](https://leetcode.cn/problems/average-time-of-process-per-machine/?envType=study-plan-v2&envId=sql-free-50)

## [join/ null](https://leetcode.cn/problems/employee-bonus/?envType=study-plan-v2&envId=sql-free-50)

## [cross join/ left join/ group by/ order by](https://leetcode.cn/problems/students-and-examinations/solutions/2366340/students-and-examinations-by-leetcode-so-3oup/?envType=study-plan-v2&envId=sql-free-50)


## [join/ having/ subquery](https://leetcode.cn/problems/managers-with-at-least-5-direct-reports/?envType=study-plan-v2&envId=sql-free-50)

## [ifnull/ 条件 avg](https://leetcode.cn/problems/confirmation-rate/solutions/2345877/xin-shou-jie-ti-fen-xi-ti-mu-yi-bu-yi-bu-8xuf/?envType=study-plan-v2&envId=sql-free-50)


# 3. 聚合函数

## [ifnull/ subquery/ left join](https://leetcode.cn/problems/average-selling-price/description/?envType=study-plan-v2&envId=sql-free-50)


## [right join](https://leetcode.cn/problems/percentage-of-users-attended-a-contest/?envType=study-plan-v2&envId=sql-free-50)


## [条件 sum/ having](https://leetcode.cn/problems/queries-quality-and-percentage/?envType=study-plan-v2&envId=sql-free-50)


## [date_format/ 条件 count/ 条件 sum](https://leetcode.cn/problems/monthly-transactions-i/?envType=study-plan-v2&envId=sql-free-50)


## [group by/ subquery](https://leetcode.cn/problems/immediate-food-delivery-ii/?envType=study-plan-v2&envId=sql-free-50)


## [subquery/ group by/ left join/ datediff/ 条件 avg](https://leetcode.cn/problems/game-play-analysis-iv/?envType=study-plan-v2&envId=sql-free-50)


# 4. 排序和分组

## [group by/ having](https://leetcode.cn/problems/sales-analysis-iii/?envType=study-plan-v2&envId=sql-free-50)






