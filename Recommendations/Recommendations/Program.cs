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

namespace Recommendations
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var (trainingDataView, testDataView) = LoadData(context);

            ITransformer model = BuildAndTrainModel(context, trainingDataView);
            EvaluateModel(context, testDataView, model);
            UseModelForSinglePrediction(context, model);
            SaveModel(context, trainingDataView.Schema, model);

            Console.ReadLine();
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
                NumberOfIterations = 20,
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
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<UserRating, UserRatingPrediction>(model);

            var testInput = new UserRating { UserId = "hanne.svard@matprat.no", ContentId = 91807 };

            var movieRatingPrediction = predictionEngine.Predict(testInput);

            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is recommended for user " + testInput.UserId + " score " + movieRatingPrediction.Score);
            }
            else
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is not recommended for user " + testInput.UserId + " score " + movieRatingPrediction.Score);
            }
        }

        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
        }

        public static void LoadModel(MLContext mlContext, string modelPath, out DataViewSchema modelSchema)
        {
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out modelSchema);
        }
    }
}

