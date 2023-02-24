export default {
    lang: 'zh-CN',
    title: 'ACM 代码模板',
    titleTemplate: false,
    description: '一份用于 ACM 类比赛的算法代码模板',
    cleanUrls: true,
    themeConfig: {
        sidebar: [
            {
                text: '',
                items: [
                    { text: '基础算法', link: '/basic-algorithm' },
                    { text: '图论', link: '/graph-theory' },
                    { text: '计算几何', link: '/computational-geometry' },
                    { text: '数据结构', link: '/data-structure' },
                    { text: '数学', link: '/math' },
                    { text: 'C++ STL', link: '/stl' },
                ],
            }
        ],
        editLink: {
            pattern: 'https://github.com/jinsuichen/CompetitiveProgrammingDocs/edit/main/docs/:path',
            text: '有错误或补充？在 GitHub 上编辑',
        },
        siteTitle: 'ACM 代码模板',
        nav: [
            { text: 'Docs', link: '/basic-algorithm' },
        ],
        socialLinks: [
            { icon: 'github', link: 'https://github.com/jinsuichen/CompetitiveProgrammingDocs' },
        ],
        docFooter: {
            prev: '上一页',
            next: '下一页',
        },
    }
}
