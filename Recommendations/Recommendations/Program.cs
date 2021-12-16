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
        public static string dataLocation = "./Data";
        public static int bookPredictionId = 34941133;

        static void Main(string[] args)
        {
            var trainingDataPath = $"{dataLocation}/ratings.csv";

            var context = new MLContext();

            //string fileName = "data.json";
            //string path = Path.Combine(@"C:\_Files\dev\hackathon\recipe-recommendations\Recommendations\Recommendations", fileName);

            //string json = "";
            //using (StreamReader r = new StreamReader(path))
            //{
            //    json = r.ReadToEnd();
            //}

            //var userRatings = JsonConvert.DeserializeObject<UserRating[]>(json);

            //foreach (var x in userRatings)
            //{
            //    Console.WriteLine($"{x.UserId} {x.ContentId}");
            //}

            var (trainingDataView, testDataView) = LoadData(context);

            ITransformer model = BuildAndTrainModel(context, trainingDataView);
            // </SnippetBuildTrainModelMain>

            // Evaluate quality of model
            // <SnippetEvaluateModelMain>
            EvaluateModel(context, testDataView, model);
            // </SnippetEvaluateModelMain>

            // Use model to try a single prediction (one row of data)
            // <SnippetUseModelMain>
            UseModelForSinglePrediction(context, model);
            // </SnippetUseModelMain>

            // Save model
            // <SnippetSaveModelMain>
            SaveModel(context, trainingDataView.Schema, model);

            Console.ReadLine();
        }


        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            string fileName = "data.json";
            string path = Path.Combine(@"C:\_Files\dev\hackathon\recipe-recommendations\Recommendations\Recommendations", fileName);

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
            // </SnippetLoadData>
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            // Add data transformations
            // <SnippetDataTransformations>
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "UserId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "contentIdEncoded", inputColumnName: "ContentId"));
            // </SnippetDataTransformations>

            // Set algorithm options and append algorithm
            // <SnippetAddAlgorithm>
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "contentIdEncoded",
                LabelColumnName = "Rating",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
            // </SnippetAddAlgorithm>

            // <SnippetFitModel>
            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);


            return model;
            // </SnippetFitModel>
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            // Evaluate model on test data & print evaluation metrics
            // <SnippetTransform>
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);
            // </SnippetTransform>

            // <SnippetEvaluate>
            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            // </SnippetEvaluate>

            // <SnippetPrintMetrics>
            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
            // </SnippetPrintMetrics>
        }

        // Use model for single prediction
        public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
        {
            // <SnippetPredictionEngine>
            Console.WriteLine("=============== Making a prediction ===============");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<UserRating, UserRatingPrediction>(model);
            // </SnippetPredictionEngine>

            // Create test input & make single prediction
            // <SnippetMakeSinglePrediction>
            var testInput = new UserRating { UserId = "hanne.svard@matprat.no", ContentId = 91807 };

            var movieRatingPrediction = predictionEngine.Predict(testInput);
            // </SnippetMakeSinglePrediction>

            // <SnippetPrintResults>
            if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is recommended for user " + testInput.UserId + " score " + movieRatingPrediction.Score);
            }
            else
            {
                Console.WriteLine("Recipe " + testInput.ContentId + " is not recommended for user " + testInput.UserId + " score " + movieRatingPrediction.Score);
            }
            // </SnippetPrintResults>
        }

        //Save model
        public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // Save the trained model to .zip file
            // <SnippetSaveModel>
            //var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");
            var modelPath = Path.Combine(@"C:\_Files\dev\hackathon\recipe-recommendations\Recommendations\Recommendations", "MovieRecommenderModel.zip");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
            // </SnippetSaveModel>
        }
    }
}

