using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Recommendations
{
    public class UserRating
    {
        //[
        //{
        //    "userId": "administrator",
        //    "userSession": null,
        //    "rating": 4,
        //    "contentId": 903616

        //},
        //{
        //    "userId": null,
        //    "userSession": "6C0BC7C2-9458-415D-9527-77D9C802E0F8",
        //    "rating": 5,
        //    "contentId": 903616
        //}
        //]

        public string UserId { get; set; }

        public string UserSession { get; set; }
        public int Rating { get; set; }
        public int ContentId { get; set; }

    }

    public class UserRatingsRoot
    {
        public IEnumerable<UserRating> UserRatings { get; set; }  
    }
    
}
