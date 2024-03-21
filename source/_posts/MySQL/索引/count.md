---
title: count
tags:
  - count
author: fbichao
categories: 
  - MySQL
  - 索引
excerpt: count
math: true
date: 2024-03-20 21:45:00
---
# 哪种 count 性能最好

![](https://file.fbichao.top/2024/03/8654d486efb53b5e4feabff5a542c42c.png)

## 1. count()

是一个聚合函数，统计符合查询条件的记录中，函数指定的参数不为 NULL 的记录有多少个。

## 2. count(主键字段)

- server 层会维护一个名叫 count 的变量
- server 层会循环向 InnoDB 读取一条记录，如果 count 函数指定的参数不为 NULL，那么就会将变量 count 加 1，直到符合查询的全部记录被读完，就退出循环。最后将 count 变量的值发送给客户端。
- 如果表里只有主键索引，没有二级索引时，那么，InnoDB 循环遍历聚簇索引，将读取到的记录返回给 server 层，然后读取记录中的 id 值，就会 id 值判断是否为 NULL，如果不为 NULL，就将 count 变量加 1。
- 如果表里有二级索引时，InnoDB 循环遍历的对象就是二级索引，相同数量的二级索引记录可以比聚簇索引记录占用更少的存储空间

## 3. count(1)

- InnoDB 循环遍历聚簇索引（主键索引），将读取到的记录返回给 server 层，但是**不会读取记录中的任何字段的值**，因为 count 函数的参数是 1，不是字段，所以不需要读取记录中的字段值。参数 1 很明显并不是 NULL，因此 server 层每从 InnoDB 读取到一条记录，就将 count 变量加 1。
- 如果表里有二级索引时，InnoDB 循环遍历的对象就二级索引了。
- 多个二级索引，选择 key_len 最小的二级索引进行扫描

## 4. count(*)

即 `count(0)`，等价于 `count(1)`

## 5. count(字段)

全表扫描

# 为什么遍历扫描

MyISAM 使用变量记录 count

InnoDB 存储引擎是支持**事务**的，同一个时刻的多个查询，由于多版本并发控制（MVCC）的原因，InnoDB 表“应该返回多少行”也是不确定的

# 优化 count(*)

## 1. 近似值

show table status 或者 explain

## 2. 额外表

想精确的获取表的记录总数，可以将这个计数值保存到单独的一张计数表中。

在新增和删除操作时，我们需要额外维护这个计数表。
