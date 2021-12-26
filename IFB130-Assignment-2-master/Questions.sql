/*Question 2 */
# Query 1
LOCK TABLES `users` WRITE;
SELECT firstName, surname, nickname, job
FROM users
WHERE suburb = 'Stafford Heights' OR suburb = 'Stafford';
UNLOCK TABLES;

# Query 2
SELECT nickname, mentorNickname 
FROM users
WHERE mentorNickname IS NOT NULL
ORDER BY mentorNickname;

# Query 3
SELECT 
    treatmentrecords.healthPracID,
    healthpractitioners.firstName,
    healthpractitioners.surname,
    count(treatmentrecords.healthPracID)
FROM
    treatmentrecords,
    healthpractitioners
WHERE
    treatmentrecords.healthPracID = healthpractitioners.healthPracID
    GROUP BY treatmentrecords.healthPracID;

#Query 4
SELECT u.firstname, u.city from users u 
WHERE u.nickname NOT IN (SELECT p.nickname from postauthors p) and u.nickname NOT IN (SELECT c.nickname from postcomments c);

#Query 5
SELECT i.illnessID, name, COUNT(datestarted) numOfUsers, MIN(datestarted) firstDate, max(datestarted) lastDate, avg(degree) 
FROM illness i
JOIN treatmentrecords tr WHERE i.illnessID=tr.illnessID
GROUP BY illnessID;

#Query 6
SELECT users.nickname, COUNT(postcomments.nickname AND postauthors.nickname) AS PostandComments
FROM users
    LEFT JOIN postauthors ON users.nickname = postauthors.nickname
        LEFT JOIN postcomments ON users.nickname = postcomments.nickname
WHERE postcomments.nickname <=1 OR postauthors.nickname <=1
GROUP BY users.nickname;

/*Question 3 */
#Query 1
INSERT INTO users (nickname, firstName, surname, birthyear)
VALUES ('stormy', 'Sam', 'Rodgers', '1982');

#Query 2
DELETE FROM phonenumber
WHERE phoneNumber LIKE '[07]%';

#Query 3
DELETE FROM phonenumber
WHERE phoneNumber LIKE '[07]%';
UPDATE healthpractitioners
SET streetNumber = '72', street = 'Evergreen_Terrace', suburb = 'Springfield'
WHERE streetNumber = '180' AND street = 'Zelda_Street' AND suburb = 'Linkburb';

/*Question 4 */
#Query 1

CREATE INDEX postindex
ON posts (postID);

#Query 2
CREATE VIEW healthy_people AS
SELECT nickname, firstName, surname, birthyear
FROM users
WHERE nickname NOT IN 
			( SELECT nickname FROM treatmentrecords);
            
/*Question 5 */
#Query 1
/*
create user 'wayne@localhost' identified by 'secret';
create user 'jake@localhost' identified by 'secret';
GRANT CREATE ON USERS TO 'jake@localhost';
GRANT DELETE ON USERS TO 'jake@localhost';
*/
GRANT CREATE ON USERS TO 'wayne@localhost';
GRANT DELETE ON USERS TO 'wayne@localhost';
REVOKE CREATE ON USERS from 'jake@localhost';
REVOKE DELETE ON USERS from 'jake@localhost';
