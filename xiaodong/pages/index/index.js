//index.js
//获取应用实例
const app = getApp()
Page({
  data: {
    // motto: 'Hello World',
    nickName:null,
    userInfo: {},
    hasUserInfo: false,
    canIUse: wx.canIUse('button.open-type.getUserInfo')
  },
  toChat: function () {
    console.log(this.data.userInfo)
    wx.navigateTo({
      url: '../contact/contact?user_id=' + this.data.userInfo.nickName
      // url: '../card/card'
    })
  },
  //事件处理函数
  bindViewTap: function() {
    wx.navigateTo({
      url: '../logs/logs'
    })
  },
  onLoad: function () {
    var that = this
    // wx.getUserInfo({
    //   success: function (res) {
    //     that.setData({
    //       nickName: res.userInfo.nickName,
    //     })
    //   },
    // })
    // if (app.globalData.userInfo) {
    //   this.setData({
    //     userInfo: app.globalData.userInfo,
    //     hasUserInfo: true
    //   })
    // } else if (this.data.canIUse){
    //   // 由于 getUserInfo 是网络请求，可能会在 Page.onLoad 之后才返回
    //   // 所以此处加入 callback 以防止这种情况
    //   app.userInfoReadyCallback = res => {
    //     this.setData({
    //       userInfo: res.userInfo,
    //       hasUserInfo: true
    //     })
    //   }
    // } else {
    //   // 在没有 open-type=getUserInfo 版本的兼容处理
    //   wx.getUserInfo({
    //     success: res => {
    //       app.globalData.userInfo = res.userInfo
    //       this.setData({
    //         userInfo: res.userInfo,
    //         hasUserInfo: true
    //       })
    //     }
    //   })
    // }
  },
  getUserInfo: function(e) {
    console.log(e.detail.userInfo);
    app.globalData.userInfo = e.detail.userInfo;
    this.setData({
      userInfo: e.detail.userInfo,
      hasUserInfo: true
    })
    this.toChat();
  },
  // shakeIcon: function (e) {
  //   console.log(e);
  //   var animation = wx.createAnimation({
  //     duration: 2000,
  //     timingFunction: 'ease',
  //   });
  //   animation.translateX(500).opacity(1).rotate(15).step()
  //   this.setData({
  //     animationData: animation.export()
  //   });
  // },
})
// setTimeout(function () {
//   wx.navigateTo({
//     url: '/pages/contact/contact'
//   })
// }, 2500)