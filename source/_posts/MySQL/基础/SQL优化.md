---
title: SQL 优化
tags:
  - SQL 优化
author: fbichao
categories: 
  - MySQL
  - 基础
excerpt: SQL 优化
math: true
date: 2024-03-21 21:45:00
---


- 插入数据：批量插入、按照主键顺序插入

- 主键优化：长度尽量短、顺序插入

- order by：通过索引返回数据

- group by：通过索引

- limit：覆盖索引和子查询

- count：count(*)

- update：根据主键/索引更新，否则可能从记录锁变成表锁

1. 建立索引
2. 建立覆盖索引
3. 根据主键/索引进行增删改，否则会从记录锁升级为表锁
4. 主键长度尽可能短，顺序插入