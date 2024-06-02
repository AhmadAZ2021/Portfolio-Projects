-- Pizza Sales Analysis 

-- Select the data from the database

SELECT * FROM pizza_sales;

-- KPI's Requirement:

-- Question 1: What is the total revenue ? 

SELECT SUM(total_price) AS Total_Revenue
FROM pizza_sales;


-- Question 2: What is the average order value ?

SELECT (SUM(total_price) / COUNT (DISTINCT order_id)) AS Average_Order_Value
FROM pizza_sales;


-- Question 3: How many pizzas have been sold ?

SELECT SUM(quantity) AS Total_Sales
FROM pizza_sales;


-- Question 4: How many orders have been placed ?
SELECT COUNT(DISTINCT order_id) AS Order_Placed
FROM pizza_sales;


-- Question 5: What is the average number of pizzas that have been sold per order ?

SELECT (CAST(SUM (quantity) AS decimal(10,2)) / 
CAST(COUNT(DISTINCT order_id) AS decimal(10,2))) AS Average_Pizzas_Per_Order
FROM pizza_sales;

-- Cast it to decimal to get more accurate result



-- Charts Requirement:

-- Daily trend for total orders:

SELECT DATENAME(DW,order_date) AS order_day , COUNT(DISTINCT order_id) AS Num_Of_Orders
FROM pizza_sales
GROUP BY DATENAME(DW,order_date);


-- Monthly trend for total orders:

SELECT DATENAME(MONTH,order_date) AS order_month , COUNT(DISTINCT order_id) AS Num_Of_Orders
FROM pizza_sales
GROUP BY DATENAME(MONTH,order_date);


-- Percentage of sales by pizza category:

SELECT pizza_category , SUM(total_price) * 100 / (SELECT SUM (total_price) FROM pizza_sales) AS Percentage_Sales
FROM pizza_sales
GROUP BY pizza_category
ORDER BY 2 DESC;

-- Percentage of sales by pizza size:
SELECT pizza_size , SUM (total_price) AS Total_Sales ,SUM(total_price) * 100 / (SELECT SUM (total_price) FROM pizza_sales) AS Percentage_Sales
FROM pizza_sales
GROUP BY pizza_size
ORDER BY 2 DESC;


-- Top 5 sellers by revenue, total quantity and total orders

SELECT TOP 5 pizza_name , SUM(Total_price) AS Total_Revenue
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 DESC;

SELECT TOP 5 pizza_name , SUM(quantity) AS Total_Quantity
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 DESC;

SELECT TOP 5 pizza_name , Count(DISTINCT order_id) AS Total_ORDERS
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 DESC;


-- Bottom 5 sellers by revenue, total quantity and total orders

SELECT TOP 5 pizza_name , SUM(Total_price) AS Total_Revenue
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 ASC;


SELECT TOP 5 pizza_name , SUM(quantity) AS Total_Quantity
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 ASC;


SELECT TOP 5 pizza_name , Count(DISTINCT order_id) AS Total_ORDERS
FROM pizza_sales
GROUP BY pizza_name
ORDER BY 2 ASC;