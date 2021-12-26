var express = require('express');
const mysql = require('mysql');
var router = express.Router();
const swaggerUI = require('swagger-ui-express');
const swaggerDocument = require('../docs/swagger.json');
var bcrypt = require('bcrypt');
const saltRounds = 10;


/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});



  
router.get("/login"), function(req, res, next) {

}

router.get("/search/:offence",function(req,res,next) {
  req.db.from('offence_columns').select('column').where("pretty", '=', req.params.offence)
  .then((rows) => {
    req.db.from('offences').where("area", "=", rows[0].area).then(areas => {
      res.json({City: areas[0]})
    })
  })
  .catch((err) => {
  console.log(err);
  res.json({"Error" : true, "Message" : "Error executing MySQL query"})
  })
 }); 

router.get("/offences", function(req,res, next){
  req.db.from('offence_columns').select("pretty")
  .then((rows) => {
    offences = [];
    res.status(200).json({"Offences" : rows.map(item => item.pretty)})
  }
  ).catch((err) => { res.status(500).json({err})})
});

router.get("/areas",function(req,res,next) {
  req.db.from('areas').select('area')
  .then(rows => {
    res.status(200).json({"Areas" : rows})
  }
  ).catch((err) => { res.status(500).json({err})})
});

router.get("/ages",function(req,res,next) {
  req.db.from('offences').select('age')
  .then(rows => {
    res.status(200).json({"Ages" : rows.map(item => item)})
  }
  ).catch((err) => { res.status(500).json({err})})
});

router.get("/genders",function(req,res,next) {
  req.db.from('offences').select('gender')
  .then(rows => {
    res.status(200).json({"Genders" : rows})
  }
  ).catch((err) => { res.status(500).json({err})})
});


router.get("/years",function(req,res,next) {
  req.db.from('offences').select('year')
  .then(rows => {
    res.status(200).json({"Genders" : rows})
  }
  ).catch((err) => { res.status(500).json({err})})
});




module.exports = router;
