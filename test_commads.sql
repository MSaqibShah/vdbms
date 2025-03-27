CREATE DATABASE ELECTRO2;
USE DATABASE ELECTRO2;
CREATE TABLE products (title text,description text,price number) EMBEDDING(description) DIMENSION 384;;


INSERT INTO products (title, description, price) VALUES ('Wireless Mouse', 'Ergonomic mouse with USB receiver and long battery life', 799);
INSERT INTO products (title, description, price) VALUES ('Mechanical Keyboard', 'Tactile keys with RGB lighting for typing enthusiasts', 2499);
INSERT INTO products (title, description, price) VALUES ('USB-C Hub', 'Connect multiple devices with one USB-C port', 1299);
INSERT INTO products (title, description, price) VALUES ('Laptop Stand', 'Aluminium stand for laptops up to 17 inches', 999);
INSERT INTO products (title, description, price) VALUES ('Noise Cancelling Headphones', 'Block distractions and enjoy deep sound quality', 3999);
SELECT * FROM products;
SELECT * FROM products WHERE price = 999 LIMIT 1;
SELECT * FROM products ORDER BY price ASC;
SELECT * FROM products WHERE title = 'Laptop Stand' ORDER BY price DESC;
SELECT * FROM products where price>99 AND description SLIKE 'Laptop stand' LIMIT 2;
UPDATE products SET price = 1799 WHERE title = 'Laptop Stand';
DELETE FROM products WHERE title = 'Wireless Mouse';
