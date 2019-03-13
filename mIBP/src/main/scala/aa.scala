import org.apache.spark.sql.SparkSession


object mIBP {
  val spark = SparkSession.builder().appName("Simple Application").master("local").getOrCreate()
  import spark.implicits._
  def main(args: Array[String]): Unit = {

  }

  def forward(data: Array[Double],probs:Array[Double],init_alpha:Map[String,Double],m_0:Double,sig:Double): Unit= {
    /*
    val data = Array(0.1,-0.9,1.2,0.3,0.2,-1.9,-0.9,1.0)
    val probs = Array(0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55)
    val m_0 = 1.5
    val sig = 3.2

     */
    val sigma = sig/Math.pow(252,0.5)
    val M = probs.length
    val states = (1 to Math.pow(2,M).toInt).map(_.toInt)
    val B_states = states.map(x => "0"*M+x.toBinaryString takeRight M)
    // val init_alpha = B_states.map(x=> (x,1.0)).toMap
    //init_alpha.map(x => ("0"*M+x._1.toBinaryString takeRight M,x._2)).filter(_._1)
    val state_vals = B_states.map(x=>(x,Math.pow(m_0,x.count(_ == '0'))*Math.pow(2-m_0,x.count(_ == '1')))).toMap
    val pa = Math.pow(2*Math.PI,-0.5)
    val likelihood = data.map(x=> state_vals.map(s => (s._1,pa/(s._2*sigma)*Math.pow(Math.E,-0.5*Math.pow((x/(s._2*sigma)),2)))).toMap)
  def one_step(probs:Array[Double],state_t1:Map[String,Double],m:Int):Map[String,Double] ={
    val T_10 = state_t1.filter(x => x._1(M-m).toString.toInt == 1).map(x=>(("0"*M + (Integer.parseInt(x._1,2)-Math.pow(2,m-1).toInt).toBinaryString) takeRight M,
      x._2*probs(m-1))).toMap
    val T_00 = state_t1.filter(x => x._1(M-m).toString.toInt == 0).map(x=>(x._1,x._2*(1-probs(m-1)))).toMap
    val V1 = (T_00.keySet ++ T_10.keySet).map {i=> (i,T_00(i) + T_10(i))}.toMap

    val T_01 = state_t1.filter(x => x._1(M-m).toString.toInt == 0).map(x=>(("0"*M + (Integer.parseInt(x._1,2)+Math.pow(2,m-1).toInt).toBinaryString) takeRight M,
      x._2*probs(m-1))).toMap
    val T_11 = state_t1.filter(x => x._1(M-m).toString.toInt == 1).map(x=>(x._1,x._2*(1-probs(m-1)))).toMap
    val V2 = (T_01.keySet ++ T_11.keySet).map {i=> (i,T_01(i) + T_11(i))}.toMap

    val ret = V1 ++ V2
    ret.map(x=> (x._1,x._2/ret.values.sum)).toMap
  }
    val alphas = (1 to data.length).foldLeft(Array(init_alpha)){(df,idx) =>
      df :+ (1 to M).foldLeft(df(idx-1)){(dat,id) => dat ++ one_step(probs,dat,id)}.map(x=> (x._1,likelihood(idx-1)(x._1)*x._2)).toMap
    }
    val betas = (1 to data.length).foldLeft(Array(B_states.map(x=>(x,1.0)).toMap)){(df,idx)=>
      df :+ (1 to M).foldLeft(df(idx-1)){(dat,id) => dat ++ one_step(probs,dat,id)}.map(x=> (x._1,likelihood(data.length - idx)(x._1)*x._2)).toMap
    }
    /*
    val state_probs = alphas.zip(betas).map(x=> (x._1.keySet ++ x._2.keySet).map {i=> (i,x._1(i) * x._2(i))}.toMap)
    (1 to data.length).foldLeft(Array(Map[String,Double])){(df,idx)=>
      df :+ alphas(idx-1).map(x=> (x._1,x._2*betas(data.length-idx).getOrElse(x._1,0.0))).toMap
    }*/
    val state_probs = alphas.zip(betas).map(x=>(x._1.keySet ++ x._2.keySet).map{i=>(i,x._1(i)*x._2(i))}.toMap)

    val ret_df = (1 to data.length-1).foldLeft(state_probs(0).map(x=>(x._1,x._2/state_probs(0).values.sum)).toSeq.toDF("states","pos_1")){(df,idx)=>
      df.join(state_probs(idx).map(x=>(x._1,x._2/state_probs(idx).values.sum)).toSeq.toDF("states","pos_"+(idx+1).toString),"states")

    }
  }
}
