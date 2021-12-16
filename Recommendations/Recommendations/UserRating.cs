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

        private string _userId;

        public string UserId
        {
            get => _userId ?? UserSession;
            set => _userId = value;
        }

        public string UserSession { get; set; }
        public float Rating { get; set; }
        public float ContentId { get; set; }

        public float Label => ContentId;

    }

    public class FloatUserRating
    {
        public float UserId { get; set; }

        public float UserSession { get; set; }
        public float Rating { get; set; }
        public float ContentId { get; set; }
    }
    
}
