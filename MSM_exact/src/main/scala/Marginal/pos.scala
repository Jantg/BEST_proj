package Marginal

import java.io._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{MultivariateGaussian, Uniform,Gaussian,Exponential,Beta,Gamma}
import scala.util.Random.nextDouble

import scala.io.Source
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
object pos {
  //val spark = SparkSession.builder()

  def main(args: Array[String]): Unit = {
    val delta_m = 0.5
    val delta_g = 0.5
    val delta_b = 5.0
    val delta_s = 2.0
    val delta_l = 2.0
    val delta_r = 0.5
    val delta_eps = 3.0

    val pr_l = 10000
    val pr_g1 = Array(50.0,500.0)
    val pr_b = Array(1.0,2.0)
    val pr_s = Array(1.0,2.0)
    val pr_r = Array(162.0,38.0)

    val kbar = Array(2,2)
    var m0_p = Uniform(1.5-delta_m,1.5+delta_m).sample(2).toArray
    var m0 = m0_p.map(v=> if (v > 2.0) 4.0-v else if (v < 1.0) 2.0-v else v)

    var gamma_1_p = Uniform(0.3-delta_g,0.3+delta_g).sample(2).toArray
    var gamma_1 = gamma_1_p.map(v=> if(v<0.0) -v else v)

    var b_p = Uniform(3.0-delta_b,3.0+delta_b).sample(2).toArray
    var b = b_p.map(v=> if(v<1.0) 2.0-v else v)

    var sig_p = Uniform(2-delta_s,2+delta_s).sample(2).toArray
    var sig = sig_p.map(v=> if(v<0.0) -v else v)
    var lambdas_p = Uniform(0.1-delta_l,0.1+delta_l).sample(kbar(0)*kbar(1)).toArray
    var lambdas = lambdas_p.map(v=> if(v<0.0) -v else v)

    var rho_p = Uniform(-0.5-delta_r,-0.5+delta_r).sample(1).toArray
    var rho = if(rho_p(0)< -1.0) math.abs(rho_p(0))-2.0 else if(rho_p(0) > 1.0) 2-rho_p(0) else rho_p(0)

    var eps_p = Uniform(0.0-delta_eps,0.0+delta_eps).sample(1).toArray
    var eps = eps_p(0)

    val prior_lamb = Exponential(pr_l)
    val prior_g1 = new Beta(pr_g1(0),pr_g1(1))
    val prior_r = new Beta(pr_r(0),pr_r(1))
    val prior_sig = Gamma(pr_s(0),pr_s(1))
    val prior_b = Gamma(pr_b(0),pr_b(1))

    var betas = Array(lambdas.slice(0,kbar(1)),lambdas.slice(kbar(1),2*kbar(1)))
    val lines = Source.fromFile("/home/jan/Downloads/msm_22_1015.txt").getLines.toArray
    val data = Array(lines.map(v=>v.split(" ")(0).toDouble),lines.map(v=>v.split(" ")(1).toDouble))
    val niter = 100000

    var betas_old = betas
    var lambdas_old = lambdas
    var m0_old = m0
    var b_old = b
    var gamma_1_old = gamma_1
    var sig_old = sig
    var rho_old = rho
    var eps_old = eps

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("pos").setMaster("local[8]")
    val sc = new SparkContext(conf)
    //val pw_lamb = new PrintWriter(new FileWriter("lamb.txt", true))
    var pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
    var pos_dist_new = pos_dist
    //var marginal = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
    //var marginal = pos_dist.values.reduce((x,y) => logSumExp(x,y))
    //var marginal_new = 0.0
    var prob = 0.0
    for( i <- 0 until niter){

      for(j<- 0 until (kbar(0))){
        /*
        if(j == kbar(0)){
          eps_p = Uniform(eps-delta_eps,eps+delta_eps).sample(1).toArray
          eps = eps_p(0)
        }else {*/
          //lambdas_p = lambdas.slice(j * kbar(1), (j + 1) * kbar(1)).map(v => math.abs(Uniform(v - delta_l, v + delta_l).sample(1)(0)))
          //lambdas_p = lambdas.slice(j*kbar(1),(j+1)*kbar(1)).map(v=> math.exp(Gaussian(math.log(v),delta_l).sample(1)(0)))
          lambdas_p = lambdas.slice(j*kbar(1),(j+1)*kbar(1)).map(v=> math.abs(v+Gaussian(0,delta_l).sample(1)(0)))
          lambdas = (lambdas_p.toList zip (j * kbar(1) until (j + 1) * kbar(1))).foldLeft(lambdas)((s, i) => s.updated(i._2, i._1))
          //betas(j) = lambdas.slice(j * kbar(1), (j + 1) * kbar(1)).zipWithIndex.foldLeft(betas(j))((s, i) => s.updated(i._2, Gaussian(0.0, math.pow(i._1,2)).sample(1)(0))))

          betas(j) = lambdas.slice(j * kbar(1), (j + 1) * kbar(1)).zipWithIndex.foldLeft(betas(j))((s, i) => s.updated(i._2, Exponential(i._1).sample(1)(0)))
          //betas = betas.updated(j,lambdas.slice(j*kbar(1),(j+1)*kbar(1)))
        //}
         pos_dist_new = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
        // prob = pos_dist.keys.map(v=>pos_dist_new(v)-pos_dist(v)).reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
        prob = pos_dist.keys.map(v=>Bigdiff(pos_dist_new(v),pos_dist(v))).reduce((x,y) => logSumExp(x,y))
         //marginal_new = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
         //if (marginal_new.isNaN||marginal.isNaN){
         //  betas = betas_old
         //  lambdas = lambdas_old
         //}
         if(math.log(nextDouble)>prob.toDouble+ //marginal_new+
                                 lambdas.map(v=>prior_lamb.logPdf(v)).sum-
                                 //lambdas.map(v=>math.log(v)).sum-lambdas_old.map(v=>math.log(v)).sum-
                                 lambdas_old.map(v=>prior_lamb.logPdf(v)).sum//-
                                 //marginal
                                     ){
           //if(j == kbar(0)){println(Array(eps,eps_old,prob.toDouble).mkString("\n"))}
           betas = betas_old
           lambdas = lambdas_old
           //eps = eps_old
         }
         betas_old = betas
         lambdas_old = lambdas
         //eps_old = eps
         pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
         //marginal = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
      }
      println(s"+++++++++++++++++++++++++++++ ${i}++++++++++++++++++++++++++")
      val pw_lamb = new PrintWriter(new FileWriter("lamb_2.txt", true))
      pw_lamb.write(lambdas.map(v=>v.toString).mkString(" "))
      pw_lamb.write(" ")
      pw_lamb.write(eps.toString)
      pw_lamb.write(" ")
      pw_lamb.write(prob.toDouble.toString)
      pw_lamb.write(" ")
      pw_lamb.write(String.format("%n"))
      pw_lamb.close()

      m0_p = m0.map(v=> Uniform(v-delta_m,v+delta_m).sample(1)(0))
      m0 = m0_p.map(v=> if (v > 2.0) 4.0-v else if (v < 1.0) 2.0-v else v)

      gamma_1_p = gamma_1.map(v=>Uniform(v-delta_g,v+delta_g).sample(1)(0))
      gamma_1 = gamma_1_p.map(v=> if(v<0.0) -v else if (v>1.0) 2.0-v else v)

      b_p = b.map(v=>Uniform(v-delta_b,v+delta_b).sample(1)(0))
      b = b_p.map(v=> if(v<1.0) 2.0-v else v)
      //b_p =  b.map(v=> math.exp(Gaussian(math.log(v),delta_b).sample(1)(0)))
      //b = b_p


      //sig_p = sig.map(v=>Uniform(v-delta_s,v+delta_s).sample(1)(0))
      //sig = sig_p.map(v=> if (v<0.0) -v else v)
      //sig_p = sig.map(v=> math.exp(Gaussian(math.log(v),delta_s).sample(1)(0)))
      //sig = sig_p
      //rho_p = Uniform(rho-delta_r,rho+delta_r).sample(1).toArray
      //rho = if(rho_p(0)< -1.0) math.abs(rho_p(0))-2.0 else if(rho_p(0) > 1.0) 2-rho_p(0) else rho_p(0)

      pos_dist_new = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
      //prob = pos_dist.keys.map(v=>pos_dist_new(v)-pos_dist(v)).reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
      prob = pos_dist.keys.map(v=>Bigdiff(pos_dist_new(v),pos_dist(v))).reduce((x,y) => logSumExp(x,y))
      //marginal_new = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
      //if (marginal_new.isNaN||marginal.isNaN){
      //  betas = betas_old
      //  lambdas = lambdas_old
      //}
      if(b.map(v=>v<1) contains true){
        m0 = m0_old
        gamma_1 = gamma_1_old
        b = b_old
        //sig = sig_old
        //rho  = rho_old
      }else if(math.log(nextDouble)>prob.toDouble+ //marginal_new+
                              b.map(v=>prior_b.logPdf(v)).sum+
                              gamma_1.map(v=>prior_g1.logPdf(v)).sum-
                              //b.map(v=>math.log(v)).sum-b_old.map(v=>math.log(v)).sum-
                              //sig.map(v=>prior_sig.logPdf(v)).sum+
                              //sig.map(v=>math.log(v)).sum-sig_old.map(v=>math.log(v)).sum-
                              b_old.map(v=>prior_b.logPdf(v)).sum -
                              gamma_1_old.map(v=>prior_g1.logPdf(v)).sum//-
                              //sig_old.map(v=>prior_sig.logPdf(v)).sum  //-
                              //marginal
                              ){
        println(Array(m0,gamma_1,b,sig,rho).deep.mkString("\n"))
          m0 = m0_old
          gamma_1 = gamma_1_old
          b = b_old
          //sig = sig_old
          //rho  = rho_old
      }//else{marginal = marginal_new}
      m0_old = m0
      gamma_1_old = gamma_1
      b_old = b
      pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
      sig_p = sig.map(v=>Uniform(v-delta_s,v+delta_s).sample(1)(0))
      sig = sig_p.map(v=> if (v<0.0) -v else v)
      //sig_p = sig.map(v=> math.exp(Gaussian(math.log(v),delta_s).sample(1)(0)))
      //sig = sig_p
      rho_p = Uniform(rho-delta_r,rho+delta_r).sample(1).toArray
      rho = if(rho_p(0)< -1.0) math.abs(rho_p(0))-2.0 else if(rho_p(0) > 1.0) 2-rho_p(0) else rho_p(0)

      pos_dist_new = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
      //prob = pos_dist.keys.map(v=>pos_dist_new(v)-pos_dist(v)).reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
      prob = pos_dist.keys.map(v=>Bigdiff(pos_dist_new(v),pos_dist(v))).reduce((x,y) => logSumExp(x,y))
      //marginal_new = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
      //if (marginal_new.isNaN||marginal.isNaN){
      //  betas = betas_old
      //  lambdas = lambdas_old
      //}
      if(math.log(nextDouble)>prob.toDouble+ //marginal_new+
        //b.map(v=>prior_b.logPdf(v)).sum+
        //gamma_1.map(v=>prior_g1.logPdf(v)).sum+
        //b.map(v=>math.log(v)).sum-b_old.map(v=>math.log(v)).sum+
       // prior_r.logPdf(math.abs(rho))+
        sig.map(v=>prior_sig.logPdf(v)).sum-
        //sig.map(v=>math.log(v)).sum-sig_old.map(v=>math.log(v)).sum-
        //b_old.map(v=>prior_b.logPdf(v)).sum -
        //gamma_1_old.map(v=>prior_g1.logPdf(v)).sum-
       // prior_r.logPdf(math.abs(rho_old))-
        sig_old.map(v=>prior_sig.logPdf(v)).sum  //-
      //marginal
      ){
        println(Array(m0,gamma_1,b,sig,rho).deep.mkString("\n"))
        //m0 = m0_old
        //gamma_1 = gamma_1_old
        //b = b_old
        sig = sig_old
        rho  = rho_old
      }//else{marginal = marginal_new}
      //m0_old = m0
      //gamma_1_old = gamma_1
      //b_old = b
      sig_old = sig
      rho_old = rho
      pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,eps,sc)
      val pw_rest = new PrintWriter(new FileWriter("rest_2.txt", true))
      pw_rest.write(m0.map(v=>v.toString).mkString(" "))
      pw_rest.write(" ")
      pw_rest.write(gamma_1.map(v=>v.toString).mkString(" "))
      pw_rest.write(" ")
      pw_rest.write(b.map(v=>v.toString).mkString(" "))
      pw_rest.write(" ")
      pw_rest.write(sig.map(v=>v.toString).mkString(" "))
      pw_rest.write(" ")
      pw_rest.write(rho.toString)
      pw_rest.write(" ")
      pw_rest.write(prob.toDouble.toString)
      pw_rest.write(" ")
      pw_rest.write(String.format("%n"))
      pw_rest.close()

      //pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,sc)
      //marginal= pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))

    }
    //val pos_dist = forward(data,m0,gamma_1,b,sig,kbar,betas,rho,sc)
    //val marginal = pos_dist.values.reduce((x,y) => math.max(x,y)+math.log(math.exp(x-math.max(x,y))+math.exp(y-math.max(x,y))))
    //val pw = new PrintWriter(new FileWriter("Marginal.txt", true))
    //pw.write(marginal.toString)
    //pw.write(String.format("%n"))
    //pw.close()
    //reduce(lambda x,y: np.max([x,y])+np.log(np.exp(x-np.max([x,y]))+np.exp(y-np.max([x,y]))),M_joint.values())

  }

  def forward(data:Array[Array[Double]], m0: Array[Double], gamma_1: Array[Double], b: Array[Double],
              sig: Array[Double], kbar: Array[Int], betas: Array[Array[Double]],
              rho: Double,eps:Double,sc:SparkContext): scala.collection.immutable.Map[(String, String),Double]  = {
    // Create Map of all possible state combinations
    val states_1 = (0 until math.pow(2, kbar(0)).toInt)
    val states_2 = (0 until math.pow(2, kbar(1)).toInt)

    // e.g. if kbar: 2,2  then (0,0),(0,1),(1,0),(1,1)
    val joint = states_1.flatMap(x => states_2.map(y => List(x, y)))

    // Turn these joint states into 8-bit binary, so the above example will be (00000000,00000000),...,(00000001,0000001)
    val states_bin = joint.map(v => v.map(x => String.format("%8s", Integer.toBinaryString((x).toByte)).replace(' ', '0')))

    // Turn the above into a Map((00000000,0000000)->whatever,....)
    val Mt = states_bin.map(v => ((v(0),v(1)),0.0)).toMap
    val return_val = (0 until data(0).length).foldLeft(Mt){(s,i)=> one_step(s,states_bin,kbar,m0,rho,sig,gamma_1,b,betas,eps,Array(data(0)(i),data(1)(i)),sc)}
    return return_val
  }

  def one_step_new(Mt:scala.collection.immutable.Map[(String, String),Double],state: IndexedSeq[List[String]],kbar: Array[Int],
               m0: Array[Double],rho:Double,sig: Array[Double], gamma_1: Array[Double],b:Array[Double],
               beta:Array[Array[Double]],eps:Double,data:Array[Double],sc:SparkContext) :scala.collection.immutable.Map[(String, String),Double]={

    // Turn all possible states in string specification into vector int specification: "1101"->Array(1,1,0,1)
    val states_vec = sc.parallelize(Mt.toList.map(v => List((v._1._1.substring(8-kbar(0),8).split("").map(_.toInt),v._1._2.substring(8-kbar(1),8).split("").map(_.toInt)))))

    // From gamma_1, b and kbar, construct array of gammas: Array(g1,g2,...gkbar)
    val gammas = (gamma_1 zip (0 until kbar.length)).map(v=> (0 until kbar(v._2)).map{vv=> if ((1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble)))>0.99999){0.99999}
    else if ((1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble)))<0.00001){0.00001}
    else{1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble))}})
    // All possible states <=> all possible switches. Record the position where switches happened.
    val switched_indexes = state.map(x => x.map(y => y.toCharArray.zipWithIndex.filter(z => z._1.toString == "1").map(v => v._2)))

    val tmp = (state zip switched_indexes).map(v=>(v._1,v._2.map(vv=>vv.toSet[Int].subsets.toList)))
    val tmp2  = tmp.map(v => (v._1,v._2(0).flatMap(_a => v._2(1).map(_b => _a -> _b))))
    val res_2 = tmp2.map(v=>v._2.map(vv=>((v._1(0),v._1(1)),(vv._1.toList.foldLeft(v._1(0).split(""))((s,i)=>s.updated(i,"0")).mkString(""),
      vv._2.toList.foldLeft(v._1(1).split(""))((s,i)=>s.updated(i,"0")).mkString("")))))
    //val res = (state zip switched_indexes).map(v => (v._1 zip v._2).map(vv=>(vv._1,vv._2.toSet[Int].subsets.flatMap(_.toList).toList.foldLeft(vv._1)((s,i) => s.updated(i,((s(i).toInt+1)%2).toString).mkString("")))))
    // List((switches,actual_switches,actual_switches in vector)

    val states_t2 = res_2.flatMap(v=>v.map(vv=> List((vv._1._1,vv._2._1,vv._2._1.substring(8-kbar(0),8).split("").map(_.toInt)),
      (vv._1._2,vv._2._2,vv._2._2.substring(8-kbar(1),8).split("").map(_.toInt))))).toList
    //val states_t1 = res_2.toList.map(v=>(v zip kbar).map(vv=>(vv._1._1,vv._1._2,vv._1._2.substring(8-vv._2,8).split("").map(_.toInt))))
    //val hoge = states_vec.map(v=>(v zip states_t1.map(vv=>vv.map(vvv=>vvv._2))).map(x=>(x._2,(x._1._1 zip x._2(0)).map(xx=>(xx._1+xx._2)%2),(x._1._2 zip x._2(1)).map(xx=>(xx._1+xx._2)%2))))

    val hoge = states_vec.map(x=> states_t2.map(v=>(v.map(vv=>vv._1),(v.map(vv=>vv._3).toList(0) zip x(0)._1).map(x=>(x._1+x._2)%2),(v.map(vv=>vv._3).toList(1) zip x(0)._2).map(x=>(x._1+x._2)%2))))

    //sometimes we get not converged exception due to the covariance matrix issues
    val lik_andswitches = hoge.map(v=>v.map(vv => (vv._2,vv._3,math.log(MultivariateGaussian(DenseVector(0.0,0.0),DenseMatrix((vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product*math.pow(sig(0),2),
      sig(0)*sig(1)*rho*math.sqrt(vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product)*math.sqrt(vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product)),(
      sig(0)*sig(1)*rho*math.sqrt(vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product)*math.sqrt(vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product),
      vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product*math.pow(sig(1),2)))).pdf(DenseVector(data))),
      (vv._1 zip kbar).map(y=>y._1.substring(8-y._2,8).split("").map(_.toInt)))))
    val switch_prob = lik_andswitches.map(v=>v.map(vv => (vv._4(1) zip (0 until vv._4(1).length)).map(vvv=> math.log(vvv._1*gammas(1)(vvv._2)/2+(1-vvv._1)*(1-gammas(1)(vvv._2)/2))).sum+
      (vv._4(0) zip (0 until vv._4(0).length)).map(vvv=> math.log(vvv._1*(sigmoid(/*eps+*/logit(gammas(0)(vvv._2))-(beta(vvv._2) zip gammas(1)).map(b=>b._1*b._2).sum+
        (beta(vvv._2) zip vv._4(1)).map(bb=>bb._1*bb._2).sum)/2)+(1-vvv._1)*(1-sigmoid(/*eps+*/logit(gammas(0)(vvv._2))-(beta(vvv._2) zip gammas(1)).map(b=>b._1*b._2).sum+
        (beta(vvv._2) zip vv._4(1)).map(bb=>bb._1*bb._2).sum)/2))).sum))
    val post = (lik_andswitches zip switch_prob).map(v=> v._1.map(vv=>(vv._1,vv._2)) zip (v._1.map(vv=>vv._3) zip v._2).map(x =>x._1+x._2))
    val aaaa = post.map(v=>v.toMap)
    //val aiue = aaaa.map(v=> v.toList.map(vv=>(String.format("%8s",vv._1._1.map(_.toString).mkString("")).replace(" ","0"),
    //  String.format("%8s",vv._1._2.map(_.toString).mkString("")).replace(" ","0"),vv._2))).map(x=>x.map(xx=>Mt.updated((xx._1,xx._2),Mt((xx._1,xx._2))+xx._3)))
    val aiue = aaaa.map(v=> v.toList.map(vv=>(String.format("%8s",vv._1._1.map(_.toString).mkString("")).replace(" ","0"),
      String.format("%8s",vv._1._2.map(_.toString).mkString("")).replace(" ","0"),vv._2))).map(x=>x.foldLeft(Mt){(s,i)=>s.updated((i._1,i._2),s((i._1,i._2))+i._3)})
    val ret_val_prep = aiue.map(v=>v.foldLeft(Mt){(s,i)=>s.updated((i._1),s(i._1)+i._2)})
    val ret_val_main = ret_val_prep.reduce((x,y)=> x ++ y.map{ case (k,v) => k -> (v + x.getOrElse(k,0.0)) })
    return ret_val_main
  }
  def one_step(Mt:scala.collection.immutable.Map[(String, String),Double],state: IndexedSeq[List[String]],kbar: Array[Int],
               m0: Array[Double],rho:Double,sig: Array[Double], gamma_1: Array[Double],b:Array[Double],
               beta:Array[Array[Double]],eps:Double,data:Array[Double],sc:SparkContext) :scala.collection.immutable.Map[(String, String),Double]={

    // Turn all possible states in string specification into vector int specification
    val states_vec = sc.parallelize(Mt.toList.map(v => List((v._1._1.substring(8-kbar(0),8).split("").map(_.toInt),v._1._2.substring(8-kbar(1),8).split("").map(_.toInt)))))

    // From gamma_1, b and kbar, construct array of gammas
    val gammas = (gamma_1 zip (0 until kbar.length)).map(v=> (0 until kbar(v._2)).map{vv=> if ((1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble)))>0.99999){0.99999}
                                                                                          else if ((1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble)))<0.00001){0.00001}
                                                                                          else{1-math.pow(1-v._1,math.pow(b(v._2),vv.toDouble))}})
    // All possible states <=> all possible switches. Record the position where switches happened.
    val switched_indexes = state.map(x => x.map(y => y.toCharArray.zipWithIndex.filter(z => z._1.toString == "1").map(v => v._2)))

    val tmp = (state zip switched_indexes).map(v=>(v._1,v._2.map(vv=>vv.toSet[Int].subsets.toList)))
    val tmp2  = tmp.map(v => (v._1,v._2(0).flatMap(_a => v._2(1).map(_b => _a -> _b))))
    val res_2 = tmp2.map(v=>v._2.map(vv=>((v._1(0),v._1(1)),(vv._1.toList.foldLeft(v._1(0).split(""))((s,i)=>s.updated(i,"0")).mkString(""),
      vv._2.toList.foldLeft(v._1(1).split(""))((s,i)=>s.updated(i,"0")).mkString("")))))
    //val res = (state zip switched_indexes).map(v => (v._1 zip v._2).map(vv=>(vv._1,vv._2.toSet[Int].subsets.flatMap(_.toList).toList.foldLeft(vv._1)((s,i) => s.updated(i,((s(i).toInt+1)%2).toString).mkString("")))))
    // List((switches,actual_switches,actual_switches in vector)

    val states_t2 = res_2.flatMap(v=>v.map(vv=> List((vv._1._1,vv._2._1,vv._2._1.substring(8-kbar(0),8).split("").map(_.toInt)),
      (vv._1._2,vv._2._2,vv._2._2.substring(8-kbar(1),8).split("").map(_.toInt))))).toList
    //val states_t1 = res_2.toList.map(v=>(v zip kbar).map(vv=>(vv._1._1,vv._1._2,vv._1._2.substring(8-vv._2,8).split("").map(_.toInt))))
    //val hoge = states_vec.map(v=>(v zip states_t1.map(vv=>vv.map(vvv=>vvv._2))).map(x=>(x._2,(x._1._1 zip x._2(0)).map(xx=>(xx._1+xx._2)%2),(x._1._2 zip x._2(1)).map(xx=>(xx._1+xx._2)%2))))

    val hoge = states_vec.map(x=> states_t2.map(v=>(v.map(vv=>vv._1),(v.map(vv=>vv._3).toList(0) zip x(0)._1).map(x=>(x._1+x._2)%2),(v.map(vv=>vv._3).toList(1) zip x(0)._2).map(x=>(x._1+x._2)%2))))

    //sometimes we get not converged exception due to the covariance matrix issues
    val lik_andswitches = hoge.map(v=>v.map(vv => (vv._2,vv._3,math.log(MultivariateGaussian(DenseVector(0.0,0.0),DenseMatrix((vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product*math.pow(sig(0),2),
      sig(0)*sig(1)*rho*math.sqrt(vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product)*math.sqrt(vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product)),(
      sig(0)*sig(1)*rho*math.sqrt(vv._2.map(x =>m0(0)*x+(2-m0(0))*(1-x)).product)*math.sqrt(vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product),
      vv._3.map(x =>m0(1)*x+(2-m0(1))*(1-x)).product*math.pow(sig(1),2)))).pdf(DenseVector(data))),
      (vv._1 zip kbar).map(y=>y._1.substring(8-y._2,8).split("").map(_.toInt)))))
    val switch_prob = lik_andswitches.map(v=>v.map(vv => (vv._4(1) zip (0 until vv._4(1).length)).map(vvv=> math.log(vvv._1*gammas(1)(vvv._2)+(1-vvv._1)*(1-gammas(1)(vvv._2)))).sum+
      (vv._4(0) zip (0 until vv._4(0).length)).map(vvv=> math.log(vvv._1*(sigmoid(/*eps+*/logit(gammas(0)(vvv._2))-(beta(vvv._2) zip gammas(1)).map(b=>b._1*b._2).sum+
        (beta(vvv._2) zip vv._4(1)).map(bb=>bb._1*bb._2).sum))+(1-vvv._1)*(1-(sigmoid(/*eps+*/logit(gammas(0)(vvv._2))-(beta(vvv._2) zip gammas(1)).map(b=>b._1*b._2).sum+
        (beta(vvv._2) zip vv._4(1)).map(bb=>bb._1*bb._2).sum))))).sum))
    val post = (lik_andswitches zip switch_prob).map(v=> v._1.map(vv=>(vv._1,vv._2)) zip (v._1.map(vv=>vv._3) zip v._2).map(x =>x._1+x._2))
    val aaaa = post.map(v=>v.toMap)
    //val aiue = aaaa.map(v=> v.toList.map(vv=>(String.format("%8s",vv._1._1.map(_.toString).mkString("")).replace(" ","0"),
    //  String.format("%8s",vv._1._2.map(_.toString).mkString("")).replace(" ","0"),vv._2))).map(x=>x.map(xx=>Mt.updated((xx._1,xx._2),Mt((xx._1,xx._2))+xx._3)))
    val aiue = aaaa.map(v=> v.toList.map(vv=>(String.format("%8s",vv._1._1.map(_.toString).mkString("")).replace(" ","0"),
      String.format("%8s",vv._1._2.map(_.toString).mkString("")).replace(" ","0"),vv._2))).map(x=>x.foldLeft(Mt){(s,i)=>s.updated((i._1,i._2),s((i._1,i._2))+i._3)})
    val ret_val_prep = aiue.map(v=>v.foldLeft(Mt){(s,i)=>s.updated((i._1),s(i._1)+i._2)})
    val ret_val_main = ret_val_prep.reduce((x,y)=> x ++ y.map{ case (k,v) => k -> (v + x.getOrElse(k,0.0)) })
    return ret_val_main
  }
  def logit(x:Double):Double={
    val ret = math.log(x/(1-x))
    return ret
  }
  def logSumExp(a : Double, b : Double) = {
    //if(a == Double.NegativeInfinity) b
    //else
    //if (b == Double.NegativeInfinity) a else
    //if(b.isNaN) a else
    //a+b;
    if(a < b) b + math.log(1 + math.exp((a-b).toDouble))
    else a + math.log(1+math.exp((b-a).toDouble));
  }
  def Bigdiff(a : Double, b : Double):Double = {
    var anew = a
    var bnew = b
    if(a == Double.NegativeInfinity){  anew =Double.MinValue}
    else if(a == Double.PositiveInfinity){ anew =Double.MaxValue}


    if(b == Double.NegativeInfinity){ bnew = Double.MinValue}
    else if(b == Double.PositiveInfinity){ bnew = Double.MaxValue}

    //else
    //if (b == Double.NegativeInfinity) a else
    //if(b.isNaN) a else
    anew-bnew;
    //if(a < b) b + BigDecimal(math.log(1 + math.exp((a-b).toDouble)))
    //else a + BigDecimal(math.log(1+math.exp((b-a).toDouble)));
  }
  def sigmoid(x:Double):Double={
    val ret = 1/(1+math.exp(-x))
    return if (ret>0.99999){0.99999}else if(ret<0.00001){0.00001} else{ret}
  }
  def repeat[A](body: => A) = new {
    def untill(condition: A => Boolean): A = {
      var a = body
      while (!condition(a)) { a = body }
      a
    }
  }
}