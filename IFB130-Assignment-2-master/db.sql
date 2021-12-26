DROP DATABASE IF EXISTS oktomook;
CREATE DATABASE  IF NOT EXISTS oktomook /*!40100 DEFAULT CHARACTER SET latin1 */;
USE `oktomook`;

CREATE TABLE `Warehouse` (
  `warehouseNumber` int NOT NULL,
  `warehouseName` varchar(255) NOT NULL,
  `streetNumber` varchar(255) NOT NULL,
  `manufacturerID` varchar(255) NOT NULL,
  `streetName` varchar(255) NOT NULL,
  `city` ENUM('Sydney', 'Brisbane', 'Melbourne') NOT NULL,
  `suburb` varchar(30) NOT NULL,
  `state` varchar(255) NOT NULL,
  `postcode` int(4) NOT NULL,
  `warehouseHeadName` varchar(30) NOT NULL,
  `numberEmployees` int DEFAULT 1,
  PRIMARY KEY (`warehouseNumber`)
);

CREATE TABLE `Publisher` (
  `publisherCode` int(30) NOT NULL,
  `publisherName` varchar(255) NOT NULL,
  `publisherCity` varchar(255) NOT NULL,
  `publisherState` varchar(255) NOT NULL,
  `publisherEmail` varchar(255),
  PRIMARY KEY (`publisherCode`)
);

CREATE TABLE `Author` (
  `authorNumber` varchar(30) NOT NULL,
  `authorName` varchar(10) NOT NULL,
  `authorEmail` varchar(30) NOT NULL,
  PRIMARY KEY (`authorNumber`)
);

CREATE TABLE `Item` (
  `itemCode` int NOT NULL,
  `itemTitle` varchar(10) NOT NULL,
  `publisherCode` varchar(30) NOT NULL,
  `itemType` varchar(30) NOT NULL,
  `stockPrice` varchar(30) NOT NULL,
  `ISBN` char(13) NOT NULL,
  PRIMARY KEY (`itemCode`)
  #FOREIGN KEY (publisherCode) REFERENCES Publisher(publisherCode)
);

CREATE TABLE `ItemWriters` (
  `itemCode` int NOT NULL,
  `authorNumber` varchar(10) NOT NULL,
  `writerSequenceNumber` varchar(30) NOT NULL,
  FOREIGN KEY (authorNumber) REFERENCES Author(authorNumber),
  FOREIGN KEY (itemCode) REFERENCES Item(itemCode)
);

CREATE TABLE `Inventory` (
  `itemCode` int NOT NULL,
  `warehouseNumber` int NOT NULL,
  `unitsOnHand` int NOT NULL,
  FOREIGN KEY (itemCode) REFERENCES Item(itemCode),
  FOREIGN KEY (warehouseNumber) REFERENCES Warehouse(warehouseNumber)
);





