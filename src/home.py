import streamlit as st
from page_setup_config import page_configure


# set up page configuration
page_configure()

st.title('Chào mừng')
st.markdown('Bạn cảm thấy **chán nản**, **căng thẳng**, **choáng ngợp**?')
st.markdown('Đừng lo, vì đã có:')
st.subheader(':balloon: :rainbow[**Nhà trị liệu ảo**] :balloon:',divider = 'rainbow')
st.balloons()
st.markdown('Không chỉ là một chatbot, Nhà trị liệu ảo còn là một hệ thống thông minh có thể phân tích cảm xúc và theo dõi chất lượng sức khỏe tâm thần của bạn.')
st.markdown('Bạn có thể trò chuyện với Nhà trị liệu ảo về bất cứ điều gì bạn nghĩ đến, chẳng hạn như vấn đề, cảm xúc, mục tiêu hoặc ước mơ của bạn. Tất cả những gì bạn cần làm là viết ra chúng ở hộp văn bản phía dưới rồi nhấn Enter. Từ đó Nhà trị liệu ảo sẽ chăm chú lắng nghe bạn và đưa ra lời khuyên hữu ích cho bạn.')
st.markdown('Thật đơn giản và dễ hiểu phải không nào?')
st.markdown('')
st.markdown('Ngoài ra, bạn cũng có thể sử dụng các biểu tượng trên thanh bên để điều chỉnh cài đặt, xem thông tin ứng dụng hoặc liên hệ với chúng tôi.')
st.markdown('')
st.markdown('Chúng tôi hy vọng bạn thích sử dụng Nhà trị liệu ảo và thấy nó có lợi cho sức khỏe của bạn.')
st.markdown('')
st.markdown('_**Hãy luôn nhớ rằng, bạn không đơn độc và chúng tôi ở đây vì bạn.**_')
st.markdown('')
st.subheader('Chức năng chính:')
st.markdown('''**Trang App** : Bạn có thể trò chuyện với chatbot và kể cho nó nghe câu chuyện của bạn. Nó sẽ giúp bạn giải quyết vấn đề của bạn''')

