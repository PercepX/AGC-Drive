<template>
    <ConfigProvider :locale="zhCN">
        <!-- <router-view></router-view> -->
        <Editor />
    </ConfigProvider>
</template>

<script setup lang="ts">
    import { ConfigProvider } from 'ant-design-vue';
    import enUS from 'ant-design-vue/es/locale/en_US';
    import zhCN from 'ant-design-vue/es/locale/zh_CN';
    // import Editor from '/@/business/chengdu/Editor.vue';
    import Editor from '/@/businessNew/Editor.vue';
</script>

<style></style>
