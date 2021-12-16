using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
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

            string fileName = "data.json";
            string path = Path.Combine(@"C:\_Files\dev\hackathon\recipe-recommendations\Recommendations\Recommendations", fileName);

            string json = "";
            using (StreamReader r = new StreamReader(path))
            {
                json = r.ReadToEnd();
            }

            var userRatings = JsonConvert.DeserializeObject<List<UserRating>>(json);

            foreach (var x in userRatings)
            {
                Console.WriteLine($"{x.UserId} {x.ContentId}");
            }

            //var reader = context.Data.TextReader(new TextLoader.Arguments()
            //{
            //    Separator = ",",
            //    HasHeader = true,
            //    Column = new[]
            //    {
            //        new TextLoader.Column("Label", DataKind.R4, 0),
            //        new TextLoader.Column("user", DataKind.R4, 1),
            //        new TextLoader.Column("bookid", DataKind.R4, 2),
            //    }
            //});

            //IDataView data = reader.Read(trainingDataPath);

            var (trainData, testData) = context.BinaryClassification.TrainTestSplit(userRatings, testFraction: 0.2);

            //var pipeline = context.Transforms.Categorical.MapValueToKey("user", "userIdEncoded")
            //    .Append(context.Transforms.Categorical.MapValueToKey("bookid", "bookIdEncoded"))
            //    .Append(new MatrixFactorizationTrainer(context, "Label", "userIdEncoded", "bookIdEncoded",
            //        advancedSettings: s => { s.NumIterations = 20; s.K = 100; }));

            //Console.WriteLine("Training recommender" + Environment.NewLine);
            //var model = pipeline.Fit(trainData);

            //var prediction = model.Transform(testData);
            //var metrics = context.Regression.Evaluate(prediction);

            //Console.WriteLine($"Model metrics: RMS - {metrics.Rms} R^2 - {metrics.RSquared}" + Environment.NewLine);

            //var predictionFunc = model.MakePredictionFunction<BookRating, BookRatingPrediction>(context);

            //var bookPrediction = predictionFunc.Predict(new BookRating
            //{
            //    user = 99,
            //    bookid = bookPredictionId
            //});

            //var bookData = LoadBookData();

            //Console.WriteLine($"Predicted rating - {Math.Round(bookPrediction.Score, 1)} for book {bookData.FirstOrDefault(b => b.BookId == bookPredictionId).BookTitle}");

            //Console.ReadLine();
        }

        //private static IList<Book> LoadBookData()
        //{
        //    var result = new List<Book>();

        //    var reader = File.OpenRead($"{dataLocation}/bookfeatures.csv");

        //    var isHeader = true;
        //    var line = String.Empty;

        //    using (var streamReader = new StreamReader(reader))
        //    {
        //        while (!streamReader.EndOfStream)
        //        {
        //            if (isHeader)
        //            {
        //                line = streamReader.ReadLine();
        //                isHeader = false;
        //            }

        //            line = streamReader.ReadLine();
        //            var data = line.Split(',');

        //            var book = new Book
        //            {
        //                BookId = int.Parse(data[3].ToString()),
        //                BookTitle = data[8].ToString(),
        //                Author = data[1].ToString(),
        //                Genre1 = data[4].ToString(),
        //                Genre2 = data[5].ToString()
        //            };

        //            result.Add(book);
        //        }
        //    }

        //    return result;
        //}
    }
}
