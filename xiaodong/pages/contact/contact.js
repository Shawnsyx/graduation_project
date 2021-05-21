// pages/contact/contact.js
const app = getApp();
var inputVal = '';
var msgList = [];
var windowWidth = wx.getSystemInfoSync().windowWidth;
var windowHeight = wx.getSystemInfoSync().windowHeight;
var keyHeight = 0;

/**
 * 初始化数据
 */
function initData(that) {
  inputVal = '';

  msgList = [
    {
    speaker: 'server',
    contentType: 'text',
    content: 'hello，你有什么问题吗？'
  }

  ]
  that.setData({
    msgList,
    inputVal
  })
}

/**
 * 计算msg总高度
 */
// function calScrollHeight(that, keyHeight) {
//   var query = wx.createSelectorQuery();
//   query.select('.scrollMsg').boundingClientRect(function(rect) {
//   }).exec();
// }

Page({

  /**
   * 页面的初始数据
   */
  data: {
    scrollHeight: '100vh',
    inputBottom: 0,
    inputval:'',
    user_id:'',
    showMyWindows:false,
    currentItem:null,
    allItem: null
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    initData(this);
    this.setData({
      user_id : options.user_id,
    });
    console.log("conatct page user_id:"+this.data.user_id)
  },

  toShowModal(e) {
    console.log(e.currentTarget.id);
    var that = this;
    this.setData({
      showMyWindows: true,
      currentItem: that.data.allItem[e.currentTarget.id]
    });
  },

  hideModal() {
    this.setData({
      showMyWindows: false
    });
  },
  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 获取聚焦
   */
  focus: function (e) {
    keyHeight = e.detail.height;
    this.setData({
      scrollHeight: (windowHeight - keyHeight) + 'px'
    });
    this.setData({
      toView: 'msg-' + (msgList.length - 1),
      inputBottom: keyHeight + 'px'
    })
    //计算msg高度
    // calScrollHeight(this, keyHeight);

  },

  //失去聚焦(软键盘消失)
  blur: function (e) {
    this.setData({
      scrollHeight: '100vh',
      inputBottom: 0
    })
  
    this.setData({
      toView: 'msg-' + (msgList.length - 1)
    })

  },
  inputFunc: function(e) {
    this.setData({
      inputval: e.detail.value
    })
  },
  // imgOnLoad(ev) {
  //   console.log(ev);
  //   let src = ev.src;
  //   scr = this.msgList[-1].content;
  // },
  /**
   * 发送点击监听
   */
  sendClick: function (e) {
    var value = this.data.inputval;
    if(value == ''){
      return;
    }
    console.log(value);
    inputVal = '';
    var inputval = '';
    this.setData({
      inputVal,
      inputval
    });
    
    msgList.push({
      speaker: 'customer',
      contentType: 'text',
      content: value
    })
    
    this.setData({
      msgList,
      inputVal
    });
    var that = this;
    wx.request({
      url: 'http://localhost:8081?question=' + value,
      method:'GET',
      header:{
        'content-type':'application/json'
      },
      success: function (res) {
        // console.log('res.data');
        // console.log(res.data['answer']);
        
        console.log(res.data)
        
        msgList.push({
          speaker: 'server',
          contentType: 'text',
          content: res.data
        })
        that.setData({
          'msgList': msgList
          })

        // for (var i = 0; i < res.data['text'].length; i++) {
        //   console.log(res.data['text'][i])
        //   // setTimeout(() => {
          
        //   // }, 0.1 * 1000);
        // }

        // if (res.data['card'].length > 0){
        //   msgList.push({
        //     speaker: 'server',
        //     contentType: 'card',
        //     content: res.data['card']
        //   })
        //   that.setData({
        //     'msgList': msgList,
        //     'allItem': res.data['card']
        //   })
        // }
      },
      fail:function(res){
        console.log("-----fail-----");
      }
    })

    // this.setData({
    //   msgList,
    //   inputVal
    // })

  },

  /**
   * 退回上一页
   */
  toBackClick: function () {
    wx.navigateBack({})
  }

})