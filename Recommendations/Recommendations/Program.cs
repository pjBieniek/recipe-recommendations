using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;
using StreamReader = System.IO.StreamReader;

namespace Recommendations
{
    class Program
    {
        static void Main(string[] args)
        {

            RunWithTraining();

            RunWithLoadingModel();

            Console.ReadLine();
        }

        public static void RunWithTraining()
        {
            var context = new MLContext();

            var (trainingDataView, testDataView) = LoadData(context);

            ITransformer model = BuildAndTrainModel(context, trainingDataView);
            EvaluateModel(context, testDataView, model);

            UseModelForSinglePrediction(context, model);
            SaveModel(context, trainingDataView.Schema, model);
        }

        public static void RunWithLoadingModel()
        {
            var context = new MLContext();

            var modelPath = Path.Combine(Environment.CurrentDirectory, "MovieRecommenderModel.zip");
            var model = LoadModel(context, modelPath, out var schema);

            MultiplePredictions(context, model);

            //UseModelForSinglePrediction(context, model);
        }

        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            string fileName = "data.json";
            string path = Path.Combine(Environment.CurrentDirectory, fileName);

            string json = "";
            using (StreamReader r = new StreamReader(path))
            {
                json = r.ReadToEnd();
            }

            // TODO: SPLIT THE DATA
            var userRatingsInMemo = JsonConvert.DeserializeObject<UserRating[]>(json);
            var splitingPoint = (int)(Math.Floor(userRatingsInMemo.Length * 0.7));
            var forTraning = userRatingsInMemo.Take(splitingPoint);
            var forTest = userRatingsInMemo.Skip(splitingPoint);

            IDataView trainingDataView = mlContext.Data.LoadFromEnumerable<UserRating>(forTraning);
            IDataView testDataView = mlContext.Data.LoadFromEnumerable<UserRating>(forTest);

            return (trainingDataView, testDataView);
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "UserId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "contentIdEncoded", inputColumnName: "ContentId"));
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "contentIdEncoded",
                LabelColumnName = "Rating",
                NumberOfIterations = 50,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);


            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("=============== Making a test prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<UserRating, UserRatingPrediction>(model);

            Console.WriteLine("=============== Checking recipe 91807 ===============");

            var testInput = new UserRating { UserId = "hanne.svard@matprat.no", ContentId = 91807 };

            var movieRatingPrediction = predictionEngine.Predict(testInput);

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is recommended for user " +
                                  testInput.UserId); //+ " score " + movieRatingPrediction.Score);
            }
            else
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is not recommended for user " +
                                  testInput.UserId); //+ " score " + movieRatingPrediction.Score);
            }
        }

        public static void MultiplePredictions(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("=============== Hanne would like these recipes the most ===============");

            string fileName = "data.json";
            string path = Path.Combine(Environment.CurrentDirectory, fileName);

            string json = "";
            using (StreamReader r = new StreamReader(path))
            {
                json = r.ReadToEnd();
            }

            var data = JsonConvert.DeserializeObject<UserRating[]>(json);

            var excludeRatedContentIds = data
                .Where(r => r.UserId == "hanne.svard@matprat.no")
                .Select(r => r.ContentId)
                .ToArray();

            var userRatingsInMemo = data
                .Where(r => !excludeRatedContentIds.Any(x => r.ContentId == x))
                .Select(r => new UserRating
                {
                    ContentId = r.ContentId,
                    Rating = default(float),
                    UserId = "hanne.svard@matprat.no", //r.UserId,
                    UserSession = r.UserSession,
                });



            var predictionEngine = mlContext.Model.CreatePredictionEngine<UserRating, UserRatingPrediction>(model);

            var movieRatingPrediction = userRatingsInMemo
                .Select(m => predictionEngine.Predict(m))
                .Where(m => !float.IsNaN(m.Score))
                .GroupBy(m => m.Label)
                .Select(g => new UserRatingPrediction { Label = g.Key, Score = g.Average(s => s.Score) })
                .OrderByDescending(m => m.Score)
                .Take(20);

            var i = 1;
            foreach (var m in movieRatingPrediction)
            {
                
                //Console.WriteLine($"{m.Label} predicted score is {m.Score}");
                Console.WriteLine($"{i}. {m.Label} link: test.matprat.no/secureUI/CMS/#context=epi.cms.contentdata:///{m.Label}");
                ++i;
            }

        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "MovieRecommenderModel.zip");

            Console.WriteLine("\n\n=============== Saving the model to a file  ===============\n\n");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public static ITransformer LoadModel(MLContext mlContext, string modelPath, out DataViewSchema modelSchema)
        {
            return mlContext.Model.Load(modelPath, out modelSchema);
        }
    }
}

