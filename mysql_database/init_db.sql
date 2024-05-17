CREATE DATABASE IF NOT EXISTS tfm;
USE tfm;

CREATE TABLE IF NOT EXISTS users (
    Id INT AUTO_INCREMENT PRIMARY KEY,
    UserName VARCHAR(255) NOT NULL,
    Password VARCHAR(255) NOT NULL,
    Rol VARCHAR(255) NOT NULL
);

-- Insertar datos de ejemplo en la tabla users
INSERT INTO users (UserName, Password, Rol) VALUES ('Alice', 'alicepass', 'edu');
INSERT INTO users (UserName, Password, Rol) VALUES ('Bob', 'bobpass', 'adm');
INSERT INTO users (UserName, Password, Rol) VALUES ('Carol', 'carolpass', 'leg');