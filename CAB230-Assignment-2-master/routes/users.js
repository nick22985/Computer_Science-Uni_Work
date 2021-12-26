var express = require('express');
var router = express.Router();

/* GET users listing. */
router.get('/', function(req, res, next) {
  res.send('respond with a resource');
});

router.post("/users/register"), function(req, res, next) {
  bcrypt.hash(req.body.password, saltRounds, function (err, has) {
    req.db.users.create ({
      email: req.body.email,
      password: hash
    })  .catch((err) => {
      console.log(err);
      res.json({"Error" : true, "Message" : "Error executing MySQL query"})
      })
    })
  }

module.exports = router;
